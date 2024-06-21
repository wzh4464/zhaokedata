###
 # File: /diffusion_coefficients.py
 # Created Date: Friday, June 21st 2024
 # Author: Zihan
 # -----
 # Last Modified: Friday, 21st June 2024 11:00:15 am
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

import numpy as np

def calculate_diffusion_coefficient(msds, time_interval):
    diffusion_coefficients = []

    for msd in msds:
        time_points = np.arange(1, len(msd) + 1) * time_interval
        slope, intercept = np.polyfit(time_points, msd, 1)
        D = slope / (2 * 3)  # 3 是维度（x, y, z）
        diffusion_coefficients.append(D)

    return diffusion_coefficients
