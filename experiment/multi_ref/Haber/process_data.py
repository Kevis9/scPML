import anndata as ann
import os

import pandas as pd

path = r'E:\YuAnHuang\kevislin\data\multi_ref\Haber'
ref_1 = ann.read(os.path.join(path, 'Haber_10x.h5ad'))
ref_2 = ann.read(os.path.join(path, 'Haber_10x_largecell.h5ad'))
ref_3 = ann.read(os.path.join(path, 'Haber_Smart-seq2.h5ad'))


query = ann.read(os.path.join(path, 'Haber_10x_region.h5ad'))


refs = [ref_1, ref_2, ref_3]
ref_datas = []
ref_labels = []
for ref in refs:
    ref_datas.append(pd.DataFrame(data=ref.X.todense(), columns=ref.var.index.tolist(), index=ref.obs.index.tolist()))
    ref_labels.append(pd.DataFrame(data=ref.obs['cell_type1'].tolist(), columns=['type']))

query_data = pd.DataFrame(data=query.X.todense(), columns=query.var.index.tolist(), index=query.obs.index.tolist())
query_label = pd.DataFrame(data=query.obs['cell_type1'].tolist(), columns=['type'])

# 先对label做一下预处理
for i in range(len(ref_labels)):
    ref_labels[i]['type'] = ref_labels[i]['type'].map(lambda x: x.split(".")[0])

query_label['type'] = query_label['type'].map(lambda x: x.split(".")[0])

# 先对ref data取细胞类型的交集
common_cell_type = list(set(ref_labels[0]['type'].tolist()) & set(ref_labels[1]['type'].tolist()) & set(ref_labels[2]['type'].tolist()))
# print(set(ref_labels[0]['type'].tolist()))
# print(set(ref_labels[1]['type'].tolist()))
# print(set(ref_labels[2]['type'].tolist()))
# print(set(query_label['type'].tolist()))
print(common_cell_type)
# print(set(common_cell_type) & set(query_label['type'].tolist()))
# exit()
idx1 = ref_labels[0]['type'].isin(common_cell_type).tolist()
idx2 = ref_labels[1]['type'].isin(common_cell_type).tolist()
idx3 = ref_labels[2]['type'].isin(common_cell_type).tolist()
idxs = [idx1, idx2, idx3]
query_idx = query_label['type'].isin(common_cell_type).tolist()

common_gene = list(set(ref_datas[0].columns.tolist()) & set(ref_datas[1].columns.tolist()) & set(ref_datas[2].columns.tolist()) & set(query_data.columns.tolist()))

for i in range(len(ref_datas)):
    ref_datas[i] = ref_datas[i][common_gene].iloc[idxs[i], :]
    ref_labels[i] = ref_labels[i].iloc[idxs[i], :]

query_data = query_data[common_gene].iloc[query_idx, :]
query_label = query_label.iloc[query_idx, :]

# save data
for i in range(len(ref_datas)):
    ref_datas[i].to_csv("raw_data/ref/data_" + str(i) + ".csv")
    ref_labels[i].to_csv("raw_data/ref/label_" + str(i) + ".csv", index=False)

query_data.to_csv("raw_data/query/data_1.csv")
query_label.to_csv("raw_data/query/label_1.csv", index=False)
