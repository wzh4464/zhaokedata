###
 # File: /levy_diffusion.py
 # Created Date: Friday, June 21st 2024
 # Author: Zihan
 # -----
 # Last Modified: Friday, 21st June 2024 11:00:31 am
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning

def levy_fit(x, a, b):
    return a * x ** b

def calculate_levy_diffusion_coefficient(msds):
    diffusion_coefficients = []

    for msd in msds:
        time_points = np.arange(1, len(msd) + 1)
        mask = ~np.isnan(msd) & (msd > 0) & (msd < 1e6)
        if np.sum(mask) > 2:
            try:
                log_time_points = np.log(time_points[mask])
                log_msd = np.log(msd[mask])
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
