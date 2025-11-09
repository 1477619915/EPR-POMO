import os.path

import torch
import json
import random
import numpy as np
import lkh
import time


def rollout(model, env, eval_type='greedy'):
    env.reset()
    actions = []
    probs = []
    reward = None
    state, reward, done = env.pre_step()
    while not done:
        cur_dist, cur_theta, xy = env.get_local_feature()
        selected, one_step_prob = model.one_step_rollout(state, cur_dist=cur_dist, cur_theta=cur_theta, xy=xy, eval_type=eval_type)
        state, reward, done = env.step(selected)
        actions.append(selected)
        probs.append(one_step_prob)

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

      # For TSPLib Euc_2D, distance must be integer.
      # A_ij = torch.round(torch.sqrt(torch.sum((points_i - points_j) ** 2, axis=-1)))
      # A_i_plus_1_j_plus_1 = torch.round(torch.sqrt(torch.sum((points_i_plus_1 - points_j_plus_1) ** 2, axis=-1)))
      # A_i_i_plus_1 = torch.round(torch.sqrt(torch.sum((points_i - points_i_plus_1) ** 2, axis=-1)))
      # A_j_j_plus_1 = torch.round(torch.sqrt(torch.sum((points_j - points_j_plus_1) ** 2, axis=-1)))

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

def check_feasible(pi):
   # input shape: (batch, multi, problem)
   pi = pi.squeeze(0)
   return (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all()

def seed_everything(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

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
    Log validation cost on 3 datasets every log steps
    '''
    self.logger['result']['val_100'].append(info[0].cpu().numpy().tolist())
    self.logger['result']['val_200'].append(info[1].cpu().numpy().tolist())
    self.logger['result']['val_500'].append(info[2].cpu().numpy().tolist())
    print(self.logger)
    directory = os.path.dirname(self.filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(self.filename, 'w') as f:
      json.dump(self.logger, f)

# 分类高质量和低质量解
def split_solutions_by_rewards(rewards, solutions, bl_val):
    higher = []
    lower = []

    for i in range(rewards.shape[0]):
        for j in range(rewards.shape[1]):
            if rewards[i, j] > bl_val[i, 0]:
                higher.append(solutions[i, j])
            elif rewards[i, j] < bl_val[i, 0]:
                lower.append(solutions[i, j])

    return higher, lower

# 截取子串及翻转
def generate_substrings_with_reverse(solution, K):
    substrings = []
    start = 0

    for i in range(len(solution)):
        if solution[i] == 0:
            if start < i:
                segment = solution[start:i]
                for j in range(len(segment) - K + 1):
                    subseq = segment[j:j+K]
                    substrings.append(subseq)
                    substrings.append(subseq[::-1])
            start = i + 1

    # 处理末尾不是 0 的情况
    if start < len(solution):
        segment = solution[start:]
        for j in range(len(segment) - K + 1):
            subseq = segment[j:j+K]
            substrings.append(subseq)
            substrings.append(subseq[::-1])

    return substrings

# 计算HGS指导幅度
def compute_cost_difference(hgs_costs, rewards, advantage, weight=1.0):
    cost_difference = hgs_costs - rewards
    cost_difference = weight / (cost_difference + 1e-6)
    mask1 = (advantage > 0) & (cost_difference > 0)
    return cost_difference, mask1

# LKH3引入
def generate_tsp_string(coords, name="instance"):
    dimension = len(coords)
    header = f"""NAME : {name}
COMMENT : {dimension}-city problem
TYPE : TSP
DIMENSION : {dimension}
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
"""
    nodes_str = ""
    for idx, (x, y) in enumerate(coords, start=1):
        nodes_str += f"{idx} {x} {y}\n"
    nodes_str += "EOF\n"
    return header + nodes_str

def extract_sections(input_string):
    lines = input_string.strip().split('\n')
    coordinates = {}
    current_section = None
    for line in lines:
        line = line.strip()
        if line.startswith("NODE_COORD_SECTION"):
            current_section = "coordinates"
        elif line.startswith("EOF"):
            break
        elif current_section == "coordinates" and line:
            parts = line.split()
            if len(parts) == 3:
                node_id, x, y = int(parts[0]), float(parts[1]), float(parts[2])
                coordinates[node_id] = (x, y)
    return coordinates

def euclidean_distance(p1, p2):
    return round(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))

def calculate_total_cost(routes, coordinates):
    total_cost = 0
    depot = coordinates[1]
    for route in routes:
        current_node = depot
        route_cost = 0
        for node in route:
            next_node = coordinates[node]
            route_cost += euclidean_distance(current_node, next_node)
            current_node = next_node
        route_cost += euclidean_distance(current_node, depot)  # Return to depot
        total_cost += route_cost
    return total_cost

def LKH_solution(batch, device, max_trials=5000, runs=30):
    """
    批量求解 TSP 问题

    参数：
        batch: numpy array，shape 为 (batch_size, n_cities, 2)
        device: 字符串，仅做占位（与LKH无关）
        max_trials: LKH最大尝试次数
        runs: LKH运行次数

    返回：
        result_list: 一个列表，每个元素是一个字典，包含 'cost' 和 'routes'
    """
    solver_path = "/home/lining/CVRP/LKH-3.0.13/LKH"  # 请确保路径正确
    result_list = []
    costs = []
    start_time = time.time()
    for i, coords in enumerate(batch):
        tsp_str = generate_tsp_string(coords, name=f"instance_{i}")
        problem = lkh.LKHProblem.parse(tsp_str)

        routes = lkh.solve(solver_path, problem=problem, max_trials=max_trials, runs=runs)

        coordinates = extract_sections(tsp_str)
        total_cost = calculate_total_cost(routes, coordinates)

        result_list.append({
            "cost": -total_cost,
            "routes": routes
        })

    costs.append(-total_cost)
    costs = torch.tensor(costs, dtype=torch.float32).view(-1, 1).to(device)
    return costs

def save_lkh3_result(file_path, i, lkh_result):
    """
    保存编号 i 的 LKH3 结果（包括 cost 和 routes）到文件。

    参数:
        file_path (str): 保存的文件路径 (.pt)
        i (int): 当前实例编号
        lkh_result (dict): 包含 "cost" 和 "routes" 的字典
    """
    try:
        existing_data = torch.load(file_path)
    except FileNotFoundError:
        existing_data = {}

    existing_data[i] = lkh_result
    torch.save(existing_data, file_path)

def load_lkh3_result(file_path, i):
    """
    加载编号 i 的 LKH3 结果（cost 和 routes）从文件中。

    参数:
        file_path (str): 保存的文件路径 (.pt)
        i (int): 实例编号

    返回:
        dict: 包含 "cost" 和 "routes" 的字典
    """
    data = torch.load(file_path)
    if i in data:
        return data[i]
    else:
        raise KeyError(f"Key {i} not found in the saved data.")