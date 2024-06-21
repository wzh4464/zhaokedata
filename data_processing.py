###
 # File: /data_processing.py
 # Created Date: Friday, June 21st 2024
 # Author: Zihan
 # -----
 # Last Modified: Friday, 21st June 2024 10:59:42 am
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

import os
import glob
import pandas as pd
import numpy as np

def load_data(datapath):
    csv_files = glob.glob(os.path.join(datapath, "*.csv"))
    data = {}

    for file in csv_files:
        filename = os.path.basename(file)
        time_point = int(filename.split('_')[1])
        df = pd.read_csv(file)
        
        for index, row in df.iterrows():
            label = row['nucleus_label']
            name = row['nucleus_name']
            if (label, name) not in data:
                data[(label, name)] = {}
            data[(label, name)][time_point] = row[['x_256', 'y_356', 'z_214', 'volume', 'surface']].values

    return data

def create_tensor(data):
    cells = list(data.keys())
    time_points = sorted({time for times in data.values() for time in times})
    num_cells = len(cells)
    num_time_points = len(time_points)
    num_features = 5  # x, y, z, volume, surface

    tensor = np.full((num_cells, num_time_points, num_features), np.nan)

    for i, cell in enumerate(cells):
        for j, time in enumerate(time_points):
            if time in data[cell]:
                tensor[i, j, :] = data[cell][time]

    return tensor, cells, time_points
