import os.path

import pandas as pd

ref_data = pd.read_csv('GSE84133_EMTAB5061/raw_data/ref/data_1.csv', index_col=0)
query_data = pd.read_csv('GSE84133_EMTAB5061/raw_data/query/data_1.csv', index_col=0)

ref_label = pd.read_csv('GSE84133_EMTAB5061/raw_data/ref/label_1.csv')
query_label = pd.read_csv('GSE84133_EMTAB5061/raw_data/query/label_1.csv')

# 这里只保留pancreas 中的主要细胞
common_type = ['alpha', 'beta', 'delta', 'gamma']
ref_idx = ref_label['type'].isin(common_type).tolist()
query_idx = query_label['type'].isin(common_type).tolist()

ref_data = ref_data.iloc[ref_idx, :]
query_data = query_data.iloc[query_idx, :]

ref_label = ref_label.iloc[ref_idx, :]
query_label = query_label.iloc[query_idx, :]

for label in common_type:
    ref_idx = (~(ref_label['type'] == label)).tolist()
    path = 'GSE84133_EMTAB5061' + label
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, 'raw_data')
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, 'ref')
    if not os.path.exists(path):
        os.makedirs(path)

    # 注意这里不要修改 ref_data和ref_label
    ref_data.iloc[ref_idx, :].to_csv(os.path.join(path, 'data_1.csv'))
    ref_label.iloc[ref_idx, :].to_csv(os.path.join(path, 'label_1.csv'), index=False)

    # 对于query_data, 注意把相关的label设置成unknown
    query_label_new = query_label['type'].replace({label:'unknown'}).to_frame()
    path = 'GSE84133_EMTAB5061' + label
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, 'raw_data')
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, 'query')
    if not os.path.exists(path):
        os.makedirs(path)
    query_label_new.to_csv(os.path.join(path, 'label_1.csv'), index=False)
    query_data.to_csv(os.path.join(path, 'data_1.csv'))