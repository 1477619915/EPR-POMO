import os
import hygese as hgs
import numpy as np
import math
import re


# 函数：读取 VRP 文件并转换为所需格式
def convert_vrp_to_dict(file_path):
    data = dict()
    coordinates = []
    demands = []

    # 读取 VRP 文件
    with open(file_path, 'r') as file:
        lines = file.readlines()
        in_node_coord = False
        in_demand_section = False

        for line in lines:
            line = line.strip()
            if line.startswith('NODE_COORD_SECTION'):
                in_node_coord = True
                continue
            elif line.startswith('DEMAND_SECTION'):
                in_node_coord = False
                in_demand_section = True
                continue
            elif line.startswith('NAME'):
                name_parts = line.split(":")
                if len(name_parts) == 2:
                    # 使用正则表达式提取 k 后面的数字
                    match = re.search(r'k(\d+)', name_parts[1].strip())
                    if match:
                        k_value = int(match.group(1))
                        data['num_vehicles'] = k_value
                continue
            elif line.startswith('CAPACITY'):
                capacity_value = int(line.split(":")[1].strip())
                data['vehicle_capacity'] = capacity_value
                continue
            elif line.startswith('EOF'):
                break

            if in_node_coord:
                parts = line.split()
                if len(parts) == 3:
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    coordinates.append((node_id, x, y))
            elif in_demand_section:
                parts = line.split()
                if len(parts) == 2:
                    demand = int(parts[1])
                    demands.append(demand)

    # 计算距离矩阵
    num_nodes = len(coordinates)
    distance_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                x1, y1 = coordinates[i][1], coordinates[i][2]
                x2, y2 = coordinates[j][1], coordinates[j][2]
                distance_matrix[i][j] = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # 将数据填入字典
    data['coordinates'] = coordinates
    data['demands'] = demands
    data['distance_matrix'] = distance_matrix.tolist()  # 转换为普通列表格式以便存储
    data['depot'] = 0
    data['service_times'] = np.zeros(len(data['demands']))

    return data


# 提取 sol 文件中的 bestcost
import re

def extract_bestcost(sol_file_path):
    bestcost = None
    with open(sol_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            # 使用正则表达式匹配以 "Cost" 开头并提取后面的数值
            match = re.match(r"^Cost\s+(\d+)", line)
            if match:
                bestcost = float(match.group(1))
                break
    return bestcost


# 设置 HGS Solver 的参数
#     nbGranular: int = 20         粒度数量
#     mu: int = 25                 群体大小
#     lambda_: int = 40            新解数量
#     nbElite: int = 4             精英个体数量
#     nbClose: int = 5
#     targetFeasible: float = 0.2  目标可行解比例
#     seed: int = 0
#     nbIter: int = 20000           迭代次数
#     timeLimit: float = 0.0        时间限制
#     useSwapStar: bool = True
ap = hgs.AlgorithmParameters(timeLimit=10, targetFeasible=0.01, nbElite=75)
hgs_solver = hgs.Solver(parameters=ap, verbose=True)

# 遍历文件夹并求解 VRP 问题
folder_path = "VRPLib/Vrp-Set-X"  # 文件夹路径，根据实际路径调整
num = 0
total_cost = 0
for filename in os.listdir(folder_path):
    if filename.endswith(".vrp"):  # 检查是否为 .vrp 文件
        file_path = os.path.join(folder_path, filename)
        sol_file_path = os.path.join(folder_path, filename.replace(".vrp", ".sol"))

        # 检查是否存在对应的 .sol 文件
        if not os.path.isfile(sol_file_path):
            print(f"No matching .sol file for {filename}")
            continue

        print(f"Processing file: {filename}")

        # 读取并转换数据
        vrp_data = convert_vrp_to_dict(file_path)

        # 提取 bestcost
        bestcost = extract_bestcost(sol_file_path)
        if bestcost is None:
            print(f"Failed to extract bestcost from {sol_file_path}")
            continue

        # 求解问题
        result = hgs_solver.solve_cvrp(vrp_data)

        # 计算并输出结果
        cost_difference_ratio = (result.cost - bestcost) / bestcost
        if cost_difference_ratio > -0.01:
            num = num + 1
            total_cost = total_cost + cost_difference_ratio

        print(f"File: {filename}")
        print(f"Cost: {result.cost}")
        print(f"Best Cost: {bestcost}")
        print(f"Routes: {result.routes}")
        print(f"Cost Difference Ratio: {cost_difference_ratio:.4f}")
        print("-" * 50)
print(f"Total_cost: {total_cost/num}")
