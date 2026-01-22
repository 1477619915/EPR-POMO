import os
import math
import torch
import numpy as np
import json
import random
import hygese as hgs

def rollout(model, env, eval_type='greedy'):
    env.reset()
    actions = []
    probs = []
    reward = None
    state, reward, done = env.pre_step()
    t = 0
    while not done:
        cur_dist, cur_theta, xy, norm_demand = env.get_cur_feature()
        selected, one_step_prob = model.one_step_rollout(state, cur_dist, cur_theta, xy, norm_demand=norm_demand, eval_type=eval_type)
        state, reward, done = env.step(selected)

        actions.append(selected)
        probs.append(one_step_prob)
        t += 1

    actions = torch.stack(actions, 1)
    if eval_type == 'greedy':
        probs = None
    else:
        probs = torch.stack(probs, 1)

    return torch.transpose(actions, 1, 2), probs, reward

def batched_two_opt_torch(cuda_points, cuda_tour, max_iterations=1000, device="cpu"):
  cuda_tour = torch.cat((cuda_tour, cuda_tour[:, 0:1]), dim=-1)
  iterator = 0
  problem_size = cuda_points.shape[0]
  with torch.inference_mode():
    batch_size = cuda_tour.shape[0]
    min_change = -1.0
    while min_change < 0.0:
      points_i = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, -1, 1, 2))
      points_j = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, 1, -1, 2))
      points_i_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, -1, 1, 2))
      points_j_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, 1, -1, 2))

      A_ij = torch.sqrt(torch.sum((points_i - points_j) ** 2, axis=-1))
      A_i_plus_1_j_plus_1 = torch.sqrt(torch.sum((points_i_plus_1 - points_j_plus_1) ** 2, axis=-1))
      A_i_i_plus_1 = torch.sqrt(torch.sum((points_i - points_i_plus_1) ** 2, axis=-1))
      A_j_j_plus_1 = torch.sqrt(torch.sum((points_j - points_j_plus_1) ** 2, axis=-1))

      change = A_ij + A_i_plus_1_j_plus_1 - A_i_i_plus_1 - A_j_j_plus_1
      valid_change = torch.triu(change, diagonal=2)

      min_change = torch.min(valid_change)
      flatten_argmin_index = torch.argmin(valid_change.reshape(batch_size, -1), dim=-1)
      min_i = torch.div(flatten_argmin_index, problem_size, rounding_mode='floor')
      min_j = torch.remainder(flatten_argmin_index, problem_size)

      if min_change < -1e-6:
        for i in range(batch_size):
          cuda_tour[i, min_i[i] + 1:min_j[i] + 1] = torch.flip(cuda_tour[i, min_i[i] + 1:min_j[i] + 1], dims=(0,))
        iterator += 1
      else:
        break

      if iterator >= max_iterations:
        break

  return cuda_tour[:, :-1]

def augment_xy_data_by_8_fold(problems):
    # problems.shape: (batch, problem, 2)
    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems


def check_feasible(pi, demand):
  # input shape: (1, multi, problem) 
  pi = pi.squeeze(0)
  multi = pi.shape[0]
  problem_size = demand.shape[1]
  demand = demand.expand(multi, problem_size)
  sorted_pi = pi.data.sort(1)[0]

  # Sorting it should give all zeros at front and then 1...n
  assert (
      torch.arange(1, problem_size + 1, out=pi.data.new()).view(1, -1).expand(multi, problem_size) ==
      sorted_pi[:, -problem_size:]
  ).all() and (sorted_pi[:, :-problem_size] == 0).all(), "Invalid tour"

  # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
  demand_with_depot = torch.cat(
      (
          torch.full_like(demand[:, :1], -1),
          demand
      ),
      1
  )
  d = demand_with_depot.gather(1, pi)

  used_cap = torch.zeros_like(demand[:, 0])
  for i in range(pi.size(1)):
      used_cap += d[:, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
      # Cannot use less than 0
      used_cap[used_cap < 0] = 0
      assert (used_cap <= 1 + 1e-4).all(), "Used more than capacity"

def seed_everything(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

class Logger(object):
  def __init__(self, filename, config):
    '''
    filename: a json file
    '''
    self.filename = filename
    self.logger = config
    self.logger['result'] = {}
    self.logger['result']['val_100'] = []
    self.logger['result']['val_200'] = []
    self.logger['result']['val_500'] = []

  def log(self, info):
    '''
    Log validation cost on 3 datasets every log step
    '''
    self.logger['result']['val_100'].append(info[0])
    self.logger['result']['val_200'].append(info[1])
    self.logger['result']['val_500'].append(info[2])

    print(self.logger)
    directory = os.path.dirname(self.filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(self.filename, 'w') as f:
        json.dump(self.logger, f)

# # 保存hgs_costs数据
# def save_hgs_costs(file_path, i, hgs_costs):
#     """
#     保存 i 和 hgs_costs 到文件中，数据以 Tensor 格式保存。
#     """
#     try:
#         # 如果文件已存在，先读取已有数据
#         existing_data = torch.load(file_path)
#     except FileNotFoundError:
#         existing_data = {}  # 文件不存在时，初始化为空字典

#     # 更新字典，将当前 i 对应的 hgs_costs 添加进去
#     existing_data[i] = hgs_costs

#     # 保存更新后的数据
#     torch.save(existing_data, file_path)

# # 加载hgs_costs数据
# def load_hgs_costs(file_path, i):
#     """
#     根据 i 从文件中加载对应的 hgs_costs，数据为 Tensor 格式。
#     """
#     data = torch.load(file_path)  # 加载整个文件内容
#     if i in data:
#         return data[i]  # 返回对应 i 的 hgs_costs
#     else:
#         raise KeyError(f"Key {i} not found in the saved data.")

def prepare_vrp_data(depot, node_coords, node_demand, num_vehicles=100, vehicle_capacity=1.0):
    data = dict()
    # 确保输入在CPU上
    if isinstance(depot, torch.Tensor): depot = depot.cpu()
    if isinstance(node_coords, torch.Tensor): node_coords = node_coords.cpu()
    if isinstance(node_demand, torch.Tensor): node_demand = node_demand.cpu()

    # 维度安全检查：确保 depot 是 [1, 2]，node_coords 是 [N, 2]
    if depot.dim() == 1:
        depot = depot.unsqueeze(0)
    if depot.dim() > 2:
        depot = depot.reshape(1, 2)

    if node_coords.dim() > 2:
        node_coords = node_coords.reshape(-1, 2)

    # 拼接坐标
    try:
        coordinates = torch.cat((depot, node_coords), dim=0)
    except RuntimeError as e:
        print(f"Error in cat: depot shape {depot.shape}, node_coords shape {node_coords.shape}")
        raise e

    # HGS通常需要需求也是列表
    demands = torch.cat((torch.tensor([0.0]), node_demand))

    # 计算距离矩阵
    coords_np = coordinates.numpy()
    num_nodes = len(coordinates)
    distance_matrix = np.zeros((num_nodes, num_nodes))

    # 简单的距离计算
    for i in range(num_nodes):
        for j in range(num_nodes):
            dist = math.sqrt((coords_np[i][0] - coords_np[j][0]) ** 2 +
                             (coords_np[i][1] - coords_np[j][1]) ** 2)
            distance_matrix[i][j] = dist

    data['coordinates'] = coordinates.tolist()
    data['demands'] = demands.tolist()
    data['distance_matrix'] = distance_matrix.tolist()
    data['depot'] = 0
    data['service_times'] = np.zeros(len(data['demands'])).tolist()
    data['num_vehicles'] = num_vehicles
    data['vehicle_capacity'] = vehicle_capacity
    return data


def hgs_solution(batch, device, num_vehicles=100, vehicle_capacity=1.0, time_limit=1.0):
    # 配置 HGS Solver
    ap = hgs.AlgorithmParameters(timeLimit=time_limit, targetFeasible=0.01)
    hgs_solver = hgs.Solver(parameters=ap, verbose=False)

    costs = []
    temp_seqs = []
    max_seq_len = 0

    batch_size = batch['depot'].size(0)

    for i in range(batch_size):
        # --- 维度修正核心部分 ---
        # 1. 提取当前实例的数据 (去掉 Batch 维度)
        depot = batch["depot"][i]  # 通常变为 [2] 或 [1, 2]
        node_coords = batch["loc"][i]  # 通常变为 [N, 2]
        node_demand = batch["demand"][i]

        # 2. 强制形状对齐
        # 确保 depot 是 [1, 2]
        if depot.dim() == 1:
            depot = depot.unsqueeze(0)  # [2] -> [1, 2]
        elif depot.dim() == 3:
            depot = depot.squeeze(0)  # [1, 1, 2] -> [1, 2]

        # 确保 node_coords 是 [N, 2]
        if node_coords.dim() == 3:
            node_coords = node_coords.squeeze(0)

        # 准备 VRP 数据
        vrp_data = prepare_vrp_data(depot, node_coords, node_demand, num_vehicles, vehicle_capacity)

        # 求解
        result = hgs_solver.solve_cvrp(vrp_data)

        # 记录 Cost
        costs.append(result.cost)

        # 线性化 Route
        flat_seq = [0]
        for route in result.routes:
            flat_seq.extend(route)
            flat_seq.append(0)

        temp_seqs.append(flat_seq)
        max_seq_len = max(max_seq_len, len(flat_seq))

    # Padding
    padded_seqs = np.zeros((batch_size, max_seq_len), dtype=np.int64)
    for i, seq in enumerate(temp_seqs):
        padded_seqs[i, :len(seq)] = seq

    # 转为 Tensor
    costs_tensor = torch.tensor(costs, dtype=torch.float32).view(-1, 1).to(device)
    seqs_tensor = torch.tensor(padded_seqs, dtype=torch.long).to(device)

    return costs_tensor, seqs_tensor


def extract_path_patterns(seq, k):
    patterns = set()
    if isinstance(seq, torch.Tensor):
        seq = seq.tolist()
    elif isinstance(seq, np.ndarray):
        seq = seq.tolist()

    seq_len = len(seq)
    for i in range(seq_len - k + 1):
        pattern = tuple(seq[i: i + k])
        patterns.add(pattern)
    return patterns


def calculate_epr_modulation(model_seqs, model_costs, elite_seqs, elite_costs, k=3, eta=0.05):
    batch_size, pomo_size, model_len = model_seqs.size()
    device = model_seqs.device

    modulation_matrix = torch.zeros((batch_size, pomo_size, model_len), device=device)
    gap_weights = torch.ones((batch_size, pomo_size), device=device)
    f_avg = model_costs.mean(dim=1, keepdim=True)

    model_seqs_cpu = model_seqs.cpu().numpy()
    elite_seqs_cpu = elite_seqs.cpu().numpy()

    for b in range(batch_size):
        e_cost = elite_costs[b].item()
        e_seq = elite_seqs_cpu[b]
        elite_patterns = extract_path_patterns(e_seq, k)

        reversed_patterns = set()
        for p in elite_patterns:
            reversed_patterns.add(tuple(reversed(p)))
        elite_patterns.update(reversed_patterns)

        for m in range(pomo_size):
            m_cost = model_costs[b, m].item()
            m_seq = model_seqs_cpu[b, m]

            delta_i = m_cost - e_cost
            if delta_i <= 1e-6: continue

            normalized_delta = delta_i / (e_cost + 1e-5)
            gap_weights[b, m] = 1.0 + normalized_delta
            delta_p = eta / (normalized_delta + 1e-2)

            is_superior = m_cost <= f_avg[b].item()

            for t in range(model_len - k + 1):
                pattern = tuple(m_seq[t: t + k])
                in_elite = pattern in elite_patterns

                if is_superior and in_elite:
                    for offset in range(k):
                        if t + offset < model_len:
                            modulation_matrix[b, m, t + offset] += delta_p
                elif (not is_superior) and (not in_elite):
                    for offset in range(k):
                        if t + offset < model_len:
                            modulation_matrix[b, m, t + offset] -= delta_p

    return modulation_matrix, gap_weights


