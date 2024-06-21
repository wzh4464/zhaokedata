###
 # File: /persistent_random_walk.py
 # Created Date: Friday, June 21st 2024
 # Author: Zihan
 # -----
 # Last Modified: Friday, 21st June 2024 11:18:23 am
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

import numpy as np

def calculate_persistent_random_walk_diffusion_coefficient(tensor, time_interval):
    diffusion_coefficients = []

    for i in range(tensor.shape[0]):
        valid_indices = ~np.isnan(tensor[i, :, 0])
        coords = tensor[i, valid_indices, :3]
        if coords.shape[0] > 1:
            displacements = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
            velocities = displacements / time_interval
            avg_velocity = np.mean(velocities)

            direction_changes = np.diff(np.arctan2(coords[1:, 1] - coords[:-1, 1], coords[1:, 0] - coords[:-1, 0]))
            avg_persistence_time = time_interval / np.mean(np.abs(direction_changes))

            D = (avg_velocity ** 2) * avg_persistence_time / 3
            diffusion_coefficients.append(D)

    return diffusion_coefficients
