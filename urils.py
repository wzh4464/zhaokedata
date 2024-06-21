###
 # File: /urils.py
 # Created Date: Friday, June 21st 2024
 # Author: Zihan
 # -----
 # Last Modified: Friday, 21st June 2024 11:00:45 am
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

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
