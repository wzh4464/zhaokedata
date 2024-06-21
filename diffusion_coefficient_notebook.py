import marimo

__generated_with = "0.6.20"
app = marimo.App()


@app.cell
def __():
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

    # 打印张量第一维度和细胞名称的对应
    print(cells[:5])

    # save to disk
    np.save(os.path.join(datapath, "tensor.npy"), tensor)
    np.save(os.path.join(datapath, "cells.npy"), cells)
    np.save(os.path.join(datapath, "time_points.npy"), time_points)
    np.save(os.path.join(datapath, "data.npy"), data)
    return (
        cell,
        cells,
        csv_files,
        data,
        datapath,
        df,
        file,
        filename,
        glob,
        i,
        index,
        j,
        label,
        name,
        np,
        num_cells,
        num_features,
        num_time_points,
        os,
        pd,
        row,
        tensor,
        time,
        time_point,
        time_points,
    )


if __name__ == "__main__":
    app.run()
