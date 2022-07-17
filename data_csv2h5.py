'''
    对给定的实验目录下的ref、query、sm目录进行文件读取，并且转为h5格式
    data的格式是cell * gene
'''

import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='[MVCC]')
parser.add_argument('--path', type=str, required=True, help='实验工作目录')
args = parser.parse_args()

data_path = os.path.join(args.path, 'data')
ref_path = os.path.join(data_path, 'ref')
query_path = os.path.join(data_path, 'query')

# model_path = os.path.join(args.path, 'model')
# result_path = os.path.join(args.path, 'result')

'''
    ref部分
'''
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
    df.to_hdf(os.path.join(args.path, 'data.h5'), 'ref_'+str(i+1)+'/data')

for i in range(len(label_file_arr)):
    df = pd.read_csv(os.path.join(ref_path, label_file_arr[i]))
    df.to_hdf(os.path.join(args.path, 'data.h5'), 'ref_'+str(i+1)+'/label')

for i in range(len(sm_file_arr)):
    df = pd.read_csv(os.path.join(ref_path, sm_file_arr[i]), index_col=0)
    # sm的命名规则为 sm_1_1 , 中间的1代表是ref_1, 后面的1代表的是view 1
    name = sm_file_arr[i]
    name = name.split('_')
    df.to_hdf(os.path.join(args.path, 'data.h5'), 'ref_'+name[1]+'/sm_'+name[2])


'''
    query 部分
'''
print("Start to process reference data")
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

print("Start to process query data")
# 读取csv数据，然后转为h5
for i in range(len(data_file_arr)):
    df = pd.read_csv(os.path.join(query_path, data_file_arr[i]), index_col=0)
    df.to_hdf(os.path.join(args.path, 'data.h5'), 'query_' + str(i + 1) + '/data')

for i in range(len(label_file_arr)):
    df = pd.read_csv(os.path.join(query_path, label_file_arr[i]))
    df.to_hdf(os.path.join(args.path, 'data.h5'), 'query_' + str(i + 1) + '/label')

for i in range(len(sm_file_arr)):
    df = pd.read_csv(os.path.join(query_path, sm_file_arr[i]), index_col=0)
    # sm的命名规则为 sm_1_1 , 中间的1代表是ref_1, 后面的1代表的是view 1
    name = sm_file_arr[i]
    name = name.split('_')
    df.to_hdf(os.path.join(args.path, 'data.h5'), 'query_' + name[1] + '/sm_' + name[2])




