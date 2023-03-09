import pandas as pd
import numpy as np
import os
'''
    1. ref和query只保留四种主要类型：alpha、beta、gaamma、delta
'''

ref_data = pd.read_csv('gsehuman_data.csv', index_col=0)
ref_label = pd.read_csv('gsehuman_label.csv')

query_data = pd.read_csv('emtab_data.csv', index_col=0)
query_label = pd.read_csv('emtab_label.csv')

cells = ['alpha', 'beta', 'gamma', 'delta']

ref_idx = ref_label.iloc[:, 0].isin(cells).tolist()
query_idx = query_label.iloc[:, 0].isin(cells).tolist()

ref_data = ref_data.iloc[ref_idx, :]
ref_label = ref_label.iloc[ref_idx, :]

query_data = query_data.iloc[query_idx, :]
query_label = query_label.iloc[query_idx, :]


'''
    将ref分成两部分，按照alpha等不同的细胞分成不同比例的两份数据
'''
def split_data(data, label, cell_dic):
    ref_1_idx = []
    ref_2_idx = []
    label_np = label.to_numpy().reshape(-1)

    for key in cell_dic.keys():
        idx = np.array(np.where(label_np == key)).squeeze().tolist()
        # print(len(idx), key)
        np.random.shuffle(idx)
        num = int(cell_dic[key] * len(idx))
        ref_1_idx += idx[:num]
        ref_2_idx += idx[num:]

    ref_1_data = data.iloc[ref_1_idx, :]
    ref_2_data = data.iloc[ref_2_idx, :]
    ref_1_label = label.iloc[ref_1_idx]
    ref_2_label = label.iloc[ref_2_idx]

    return (ref_1_data, ref_1_label, ref_2_data, ref_2_label)


# alpha
cell_ref = {
    'alpha': 0.5,
    'beta' : 0.5,
    'gamma': 0.5,
    'delta': 0.5,
}
dir = 'alpha'


ref_1_data, ref_1_label, ref_2_data, ref_2_label = split_data(ref_data, ref_label, cell_ref)
if not os.path.exists(dir):
    os.makedirs(dir)

ref_1_data.to_csv(os.path.join(dir, 'ref_1_data.csv'))
ref_1_label.to_csv(os.path.join(dir, 'ref_1_label.csv'), index=False)
ref_2_data.to_csv(os.path.join(dir, 'ref_2_data.csv'))
ref_2_label.to_csv(os.path.join(dir, 'ref_2_label.csv'), index=False)
query_data.to_csv(os.path.join(dir, 'query_data.csv'))
query_label.to_csv(os.path.join(dir, 'query_label.csv'), index=False)

exit()
#beta
cell_ref = {
    'alpha': 2500,
    'beta': 20,
    'gamma': 230,
    'delta': 580,
}
dir = 'beta'
ref_1_data, ref_1_label, ref_2_data, ref_2_label = split_data(ref_data, ref_label, cell_ref)
if not os.path.exists(dir):
    os.makedirs(dir)

ref_1_data.to_csv(os.path.join(dir, 'ref_1_data.csv'))
ref_1_label.to_csv(os.path.join(dir, 'ref_1_label.csv'), index=False)
ref_2_data.to_csv(os.path.join(dir, 'ref_2_data.csv'))
ref_2_label.to_csv(os.path.join(dir, 'ref_2_label.csv'), index=False)

query_data.to_csv(os.path.join(dir, 'query_data.csv'))
query_label.to_csv(os.path.join(dir, 'query_label.csv'), index=False)

# gamma
cell_ref = {
    'alpha': 2500,
    'beta': 2500,
    'gamma': 20,
    'delta': 580,
}
dir = 'gamma'
ref_1_data, ref_1_label, ref_2_data, ref_2_label = split_data(ref_data, ref_label, cell_ref)
if not os.path.exists(dir):
    os.makedirs(dir)

ref_1_data.to_csv(os.path.join(dir, 'ref_1_data.csv'))
ref_1_label.to_csv(os.path.join(dir, 'ref_1_label.csv'), index=False)
ref_2_data.to_csv(os.path.join(dir, 'ref_2_data.csv'))
ref_2_label.to_csv(os.path.join(dir, 'ref_2_label.csv'), index=False)

query_data.to_csv(os.path.join(dir, 'query_data.csv'))
query_label.to_csv(os.path.join(dir, 'query_label.csv'), index=False)


# delta
cell_ref = {
    'alpha': 2500,
    'beta': 2500,
    'gamma': 230,
    'delta': 20,
}
dir = 'delta'
ref_1_data, ref_1_label, ref_2_data, ref_2_label = split_data(ref_data, ref_label, cell_ref)
if not os.path.exists(dir):
    os.makedirs(dir)

ref_1_data.to_csv(os.path.join(dir, 'ref_1_data.csv'))
ref_1_label.to_csv(os.path.join(dir, 'ref_1_label.csv'), index=False)
ref_2_data.to_csv(os.path.join(dir, 'ref_2_data.csv'))
ref_2_label.to_csv(os.path.join(dir, 'ref_2_label.csv'), index=False)
query_data.to_csv(os.path.join(dir, 'query_data.csv'))
query_label.to_csv(os.path.join(dir, 'query_label.csv'), index=False)