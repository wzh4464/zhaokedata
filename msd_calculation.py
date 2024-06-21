###
 # File: /msd_calculation.py
 # Created Date: Friday, June 21st 2024
 # Author: Zihan
 # -----
 # Last Modified: Friday, 21st June 2024 10:59:57 am
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

import numpy as np

def calculate_msd(tensor):
    num_cells, num_time_points, _ = tensor.shape
    msds = []

    for i in range(num_cells):
        valid_indices = ~np.isnan(tensor[i, :, 0])
        coords = tensor[i, valid_indices, :3]
        
        if coords.shape[0] > 1:
            time_diffs = np.arange(1, coords.shape[0])
            msd = np.zeros_like(time_diffs, dtype=np.float64)

            for t in time_diffs:
                diffs = coords[t:] - coords[:-t]
                squared_diffs = np.sum(diffs ** 2, axis=1)
                msd[t-1] = np.mean(squared_diffs)
            
            msds.append(msd)

    return msds
