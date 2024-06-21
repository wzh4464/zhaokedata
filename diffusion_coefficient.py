###
 # File: /diffusion_coefficient.py
 # Created Date: Friday, June 21st 2024
 # Author: Zihan
 # -----
 # Last Modified: Friday, 21st June 2024 10:24:28 am
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

# %%
import numpy as np
import pandas as pd
import glob
import os

# 获取所有 CSV 文件的路径
datapath = "/home/zihan/dataset/zhaokedata/200113plc1p2"
csv_files = glob.glob(os.path.join(datapath, "*.csv"))

# 初始化一个字典用于存储所有数据
data = {}

# 读取所有 CSV 文件，并将时间信息添加到字典中
for file in csv_files:
    # 从文件名中提取时间信息
    filename = os.path.basename(file)
    time_point = int(filename.split('_')[1])
    
    # 读取 CSV 文件
    df = pd.read_csv(file)
    
    # 添加数据到字典中
    for index, row in df.iterrows():
        label = row['nucleus_label']
        name = row['nucleus_name']
        if (label, name) not in data:
            data[(label, name)] = {}
        data[(label, name)][time_point] = row[['x_256', 'y_356', 'z_214', 'volume', 'surface']].values

# 获取所有的细胞和时间点
cells = list(data.keys())
time_points = sorted({time for times in data.values() for time in times})

# 初始化张量，维度为 (num_cells, num_time_points, num_features)
num_cells = len(cells)
num_time_points = len(time_points)
num_features = 5  # x, y, z, volume, surface
tensor = np.full((num_cells, num_time_points, num_features), np.nan)

# 填充张量
for i, cell in enumerate(cells):
    for j, time in enumerate(time_points):
        if time in data[cell]:
            tensor[i, j, :] = data[cell][time]

# 打印张量形状
print(f"Tensor shape: {tensor.shape}")

# 示例输出部分张量数据
print(tensor[:5, :5, :])

# 打印张量第一维度和细胞名称的对应
print(cells[:5])


# %%
import numpy as np

def find_subtensors(tensor):
    depth, rows, cols = tensor.shape
    visited = np.zeros_like(tensor, dtype=bool)
    subtensors = []

    def find_max_subtensor(d, r, c):
        max_d, max_r, max_c = d, r, c
        while max_d < depth and not np.isnan(tensor[max_d, r, c]) and not visited[max_d, r, c]:
            max_r_temp = r
            while max_r_temp < rows and not np.isnan(tensor[max_d, max_r_temp, c]) and not visited[max_d, max_r_temp, c]:
                sum_c_temp = c
                while sum_c_temp < cols and not np.isnan(tensor[max_d, max_r_temp, sum_c_temp]) and not visited[max_d, max_r_temp, sum_c_temp]:
                    sum_c_temp += 1
                max_r_temp += 1
                max_c = max(max_c, sum_c_temp)
            max_d += 1
            max_r = max(max_r, max_r_temp)
        return (d, r, c, max_d - 1, max_r - 1, max_c - 1)

    for d in range(depth):
        for i in range(rows):
            for j in range(cols):
                if not np.isnan(tensor[d, i, j]) and not visited[d, i, j]:
                    d1, r1, c1, d2, r2, c2 = find_max_subtensor(d, i, j)
                    subtensors.append(((d1, r1, c1), (d2, r2, c2)))
                    for d_ in range(d1, d2 + 1):
                        for r in range(r1, r2 + 1):
                            for c in range(c1, c2 + 1):
                                visited[d_, r, c] = True

    return subtensors

# 示例三维张量
# tensor = np.array([
#     [
#         [1, 2, 3, np.nan, 5],
#         [6, np.nan, 8, 9, 10],
#         [np.nan, np.nan, 11, 12, 13],
#         [14, 15, 16, np.nan, np.nan]
#     ],
#     [
#         [17, 18, np.nan, 20, 21],
#         [np.nan, 23, 24, 25, 26],
#         [27, 28, 29, 30, np.nan],
#         [32, 33, 34, 35, 36]
#     ]
# ])

subtensors = find_subtensors(tensor)
for subtensor in subtensors:
    print(f"Subtensor from {subtensor[0]} to {subtensor[1]}")


# %%
import numpy as np

def calculate_msd(tensor):
    num_cells, num_time_points, _ = tensor.shape
    msds = []

    for i in range(num_cells):
        # 提取当前细胞的坐标数据，忽略 NaN 值
        valid_indices = ~np.isnan(tensor[i, :, 0])
        coords = tensor[i, valid_indices, :3]
        
        if coords.shape[0] > 1:
            # 计算时间差
            time_diffs = np.arange(1, coords.shape[0])
            msd = np.zeros_like(time_diffs, dtype=np.float64)

            for t in time_diffs:
                diffs = coords[t:] - coords[:-t]
                squared_diffs = np.sum(diffs ** 2, axis=1)
                msd[t-1] = np.mean(squared_diffs)
            
            msds.append(msd)

    return msds

# 计算 MSD
msds = calculate_msd(tensor)

# 打印 MSD 示例
for msd in msds[:3]:
    print(msd)


# %%
# brownian motion
def calculate_diffusion_coefficient(msds, time_interval):
    diffusion_coefficients = []

    for msd in msds:
        # 线性拟合 MSD 曲线
        time_points = np.arange(1, len(msd) + 1) * time_interval
        slope, intercept = np.polyfit(time_points, msd, 1)
        D = slope / (2 * 3)  # 3 是维度（x, y, z）
        diffusion_coefficients.append(D)

    return diffusion_coefficients

# 假设时间间隔为 1 单位时间
time_interval = 1

# 计算扩散系数
diffusion_coefficients = calculate_diffusion_coefficient(msds, time_interval)

# 打印扩散系数示例
print(diffusion_coefficients[:3])


# %%
def calculate_persistent_random_walk_diffusion_coefficient(tensor):
    diffusion_coefficients = []

    for i in range(tensor.shape[0]):
        valid_indices = ~np.isnan(tensor[i, :, 0])
        coords = tensor[i, valid_indices, :3]
        if coords.shape[0] > 1:
            # 计算速度
            displacements = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
            velocities = displacements / time_interval
            avg_velocity = np.mean(velocities)

            # 计算持久时间
            direction_changes = np.diff(np.arctan2(coords[1:, 1] - coords[:-1, 1], coords[1:, 0] - coords[:-1, 0]))
            avg_persistence_time = time_interval / np.mean(np.abs(direction_changes))

            # 计算扩散系数
            D = (avg_velocity ** 2) * avg_persistence_time / 3
            diffusion_coefficients.append(D)

    return diffusion_coefficients

# 计算持久随机游动扩散系数
persistent_rw_diffusion_coefficients = calculate_persistent_random_walk_diffusion_coefficient(tensor)

# 打印持久随机游动扩散系数示例
print(persistent_rw_diffusion_coefficients[:3])


# %%
from scipy.optimize import curve_fit
import numpy as np
from scipy.optimize import OptimizeWarning

def levy_fit(x, a, b):
    return a * x ** b

def calculate_levy_diffusion_coefficient(msds):
    diffusion_coefficients = []

    for msd in msds:
        time_points = np.arange(1, len(msd) + 1)
        # 去除 NaN 值和异常值
        mask = ~np.isnan(msd) & (msd > 0) & (msd < 1e6)
        if np.sum(mask) > 2:  # 至少需要三个点进行拟合
            try:
                # 对数据进行对数转换
                log_time_points = np.log(time_points[mask])
                log_msd = np.log(msd[mask])
                
                # 提供初始参数并增加最大函数调用次数
                popt, pcov = curve_fit(levy_fit, log_time_points, log_msd, p0=[1, 0.5], maxfev=200000000)
                
                alpha = popt[1]
                D = np.exp(popt[0]) ** (2 / alpha)
                diffusion_coefficients.append(D)
            except (RuntimeError, OverflowError, OptimizeWarning) as e:
                print(f"Error fitting Lévy flight model for MSD: {e}")
                diffusion_coefficients.append(np.nan)
        else:
            diffusion_coefficients.append(np.nan)

    return diffusion_coefficients

# 计算 Lévy 飞行扩散系数
levy_diffusion_coefficients = calculate_levy_diffusion_coefficient(msds)
levy_diffusion_coefficients = np.log(levy_diffusion_coefficients)

# inf -> 1e5
levy_diffusion_coefficients[np.isinf(levy_diffusion_coefficients)] = 1e4

# 打印 Lévy 飞行扩散系数示例
print(levy_diffusion_coefficients[:300])


# %%
# 扩散矩阵
# normalize with max value

print("maxes:")
print(np.max(diffusion_coefficients))
print(np.max(persistent_rw_diffusion_coefficients))
print(np.max(levy_diffusion_coefficients))

# if has nan, record the index
nan_index = np.isnan(diffusion_coefficients) | np.isnan(persistent_rw_diffusion_coefficients) | np.isnan(levy_diffusion_coefficients)
not_nan_index = ~nan_index

# print :30
print(f"nan_index: {nan_index[:30]}")
print(f"not_nan_index: {not_nan_index[:30]}")

diffusion_matrix = np.array([
    diffusion_coefficients,
    persistent_rw_diffusion_coefficients,
    levy_diffusion_coefficients
])

# remove nan_index and form new diffusion matrix
i = 0
for index in range(len(nan_index)):
    if not nan_index[index]:
        i += 1
    else:
        diffusion_matrix = np.delete(diffusion_matrix, i, axis=1)

for k in range(diffusion_matrix.shape[1]):
    diffusion_matrix[:, k] = diffusion_matrix[:, k] / np.max(diffusion_matrix[:, k])

# diffusion_matrix = np.array([
#     diffusion_coefficients/np.max(diffusion_coefficients),
#     persistent_rw_diffusion_coefficients/np.max(persistent_rw_diffusion_coefficients),
#     levy_diffusion_coefficients/np.max(levy_diffusion_coefficients)
# ])

print(f"Diffusion matrix 3x3: {diffusion_matrix[:3, :3]}")

# do coclustering
from cocluster import CoCluster
import numpy as np
import matplotlib.pyplot as plt

# 初始化 CoCluster 模型
ks = [2, 3]
for k in ks:
    model = CoCluster(diffusion_matrix, k)
    # _, label_u_1, label_v_1 = model.co_cluster()
    _, S1, _ = model.co_SVD()

plt.figure(figsize=(10, 6))
plt.plot(S1, ks, marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("SVD objective")
plt.title("SVD objective vs. number of clusters")
plt.grid()
plt.show()



