"""
    对给定的实验目录下的ref、query、sm目录进行文件读取，并且转为h5格式
    data的格式是cell * gene
"""

import os
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='[MVCC]')
parser.add_argument('--path', type=str, required=False, help='实验工作目录')
args = parser.parse_args()

# args.path = r'..\experiment\multi_ref\MCA_liver'
data_path = os.path.join(args.path, 'data')
# 暂时改成raw data
data_path = os.path.join(args.path, 'raw_data')
ref_path = os.path.join(data_path, 'ref')
query_path = os.path.join(data_path, 'query')


# model_path = os.path.join(args.path, 'MVCC')
# result_path = os.path.join(args.path, 'result')

'''
    ref部分
'''
print("path is "+args.path)
print("Start to process ref data")
assert len(os.listdir(ref_path)) != 0
data_file_arr = []
label_file_arr = []
sm_file_arr = []
for file in os.listdir(ref_path):
    if 'data' in file.split('_'):
        data_file_arr.append(file)
    elif 'label' in file.split('_'):
        label_file_arr.append(file)
    elif 'sm' in file.split('_'):
        sm_file_arr.append(file)

data_file_arr.sort()
label_file_arr.sort()
sm_file_arr.sort()

# 读取csv数据，然后转为h5
for i in range(len(data_file_arr)):
    df = pd.read_csv(os.path.join(ref_path, data_file_arr[i]), index_col=0)
    df.to_hdf(os.path.join(args.path, 'data.h5'), 'ref_'+data_file_arr[i].split('.')[0].split('_')[1]+'/data')

for i in range(len(label_file_arr)):
    df = pd.read_csv(os.path.join(ref_path, label_file_arr[i]))
    df.to_hdf(os.path.join(args.path, 'data.h5'), 'ref_'+label_file_arr[i].split('.')[0].split('_')[1]+'/label')


for i in range(len(sm_file_arr)):
    print(os.path.join(ref_path, sm_file_arr[i]))
    df = pd.read_csv(os.path.join(ref_path, sm_file_arr[i]), index_col=0)
    # sm的命名规则为 sm_1_1.csv , 中间的1代表是ref_1, 后面的1代表的是view 1
    name = sm_file_arr[i]
    name = name.split('.')[0].split('_')
    df.to_hdf(os.path.join(args.path, 'data.h5'), 'ref_'+name[1]+'/sm_'+name[2])


'''
    query
'''
print("Start to process query data")
assert len(os.listdir(query_path)) != 0
data_file_arr = []
label_file_arr = []
sm_file_arr = []
for file in os.listdir(query_path):
    if 'data' in file.split('_'):
        data_file_arr.append(file)
    elif 'label' in file.split('_'):
        label_file_arr.append(file)
    elif 'sm' in file.split('_'):
        sm_file_arr.append(file)

data_file_arr.sort()
label_file_arr.sort()
sm_file_arr.sort()

# 读取csv数据，然后转为h5
for i in range(len(data_file_arr)):
    df = pd.read_csv(os.path.join(query_path, data_file_arr[i]), index_col=0)
    df.to_hdf(os.path.join(args.path, 'data.h5'), 'query_' + data_file_arr[i].split('.')[0].split('_')[1] + '/data')

for i in range(len(label_file_arr)):
    df = pd.read_csv(os.path.join(query_path, label_file_arr[i]))
    df.to_hdf(os.path.join(args.path, 'data.h5'), 'query_' + label_file_arr[i].split('.')[0].split('_')[1] + '/label')

for i in range(len(sm_file_arr)):
    df = pd.read_csv(os.path.join(query_path, sm_file_arr[i]), index_col=0)
    # sm的命名规则为 sm_1_1 , 中间的1代表是ref_1, 后面的1代表的是view 1
    print("query sm shape ", df.shape)
    name = sm_file_arr[i]
    name = name.split('.')[0].split('_')
    df.to_hdf(os.path.join(args.path, 'data.h5'), 'query_' + name[1] + '/sm_' + name[2])


import h5py
print("data.h5中的keys为:")
f = h5py.File(os.path.join(args.path,'data.h5'), 'r')

def print_attrs(name, obj):
    if isinstance(obj, h5py.Dataset):
        pass
    else:
        print(name)

f.visititems(print_attrs)