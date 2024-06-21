###
 # File: /main.py
 # Created Date: Tuesday, June 18th 2024
 # Author: Zihan
 # -----
 # Last Modified: Friday, 21st June 2024 11:19:22 am
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

# %%
import numpy as np
import pandas as pd
from data_processing import load_data, create_tensor
from msd_calculation import calculate_msd
from diffusion_coefficients import calculate_diffusion_coefficient
from levy_diffusion import calculate_levy_diffusion_coefficient
from persistent_random_walk import calculate_persistent_random_walk_diffusion_coefficient

# # 设置数据路径
# raw_datapath = "/home/zihan/dataset/zhaokedata/200113plc1p2"

# # 加载数据并创建张量
# data = load_data(raw_datapath)
# tensor, cells, time_points = create_tensor(data)

# # 保存张量和细胞信息
# local_datapath = "./data/"
# np.save(f"{local_datapath}tensor.npy", tensor)
# np.save(f"{local_datapath}cells.npy", cells)
# np.save(f"{local_datapath}time_points.npy", time_points)

# 读取张量和细胞信息
local_datapath = "./data/"
# 读取张量和细胞信息
tensor = np.load(f"{local_datapath}tensor.npy")
cells = np.load(f"{local_datapath}cells.npy")
time_points = np.load(f"{local_datapath}time_points.npy")

# 计算 MSD
msds = calculate_msd(tensor)

# 计算扩散系数
time_interval = 1
diffusion_coefficients = calculate_diffusion_coefficient(msds, time_interval)
levy_diffusion_coefficients = calculate_levy_diffusion_coefficient(msds)
persistent_rw_diffusion_coefficients = calculate_persistent_random_walk_diffusion_coefficient(tensor, time_interval)

# 打印结果
print(f"Diffusion coefficients: {diffusion_coefficients[:3]}")
print(f"Lévy diffusion coefficients: {levy_diffusion_coefficients[:3]}")
print(f"Persistent random walk diffusion coefficients: {persistent_rw_diffusion_coefficients[:3]}")
