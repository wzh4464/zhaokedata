###
 # File: /main.py
 # Created Date: Tuesday, June 18th 2024
 # Author: Zihan
 # -----
 # Last Modified: Tuesday, 18th June 2024 1:16:10 pm
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

# %%
import numpy as np
import os
import logging

HOME = os.environ['HOME'] + '/'
datapath = f'{HOME}dataset/zhaokedata'
file_list = os.listdir(datapath)
# remove all '._' files
file_list = [f for f in file_list if not f.startswith('._')]

logging.basicConfig(level=logging.INFO)
logging.info(f'file_list: {file_list}')

# INFO:root:file_list: ['200113plc1p2.npy', '200113plc1p2_TD.npy', '200323plc1p1_Names.npy', '200113plc1p2_Names.npy', '200323plc1p1_TD.npy']

# get size of each file
for f in file_list:
    data = np.load(f'{datapath}/{f}', allow_pickle=True)
    logging.info(f'{f}: {data.shape}')


# %%
data = np.load(
    '/home/zihan/codes/zhaokedata/200113plc1p21.npy', allow_pickle=True
)

# print data as string
print(data)

# %%
# 打印data的类型
print("Type of data:", type(data))

# 打印常用属性
print("Number of dimensions (ndim):", data.ndim)
print("Shape of the array (shape):", data.shape)
print("Total number of elements (size):", data.size)
print("Data type of elements (dtype):", data.dtype)
print("Size of each element (itemsize):", data.itemsize, "bytes")
print("Total bytes of the array (nbytes):", data.nbytes, "bytes")
print("Transpose of the array (T):\n", data.T)
print("Real part of the array (real):\n", data.real)
print("Imaginary part of the array (imag):\n", data.imag)

# 打印数组的元素
print("Flat iterator of the array (flat):", list(data.flat))

# %%



