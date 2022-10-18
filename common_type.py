import os
import pandas as pd

ref_save_path = 'experiment/species/gse_emtab_common_type/mouse_human/data/ref'
query_save_path = 'experiment/species/gse_emtab_common_type/mouse_human/data/query'

ref_path = 'E:/yuanhuang/kevislin/data/species/EMTAB5061_GSE/mouse'

# query1_path = 'H:/yuanhuang/kevislin/data/platform/InDrop'
# query2_path = 'H:/yuanhuang/kevislin/data/platform/InDrop'
# query3_path = 'H:/yuanhuang/kevislin/data/platform/Seq_Well'
# query4_path = 'H:/yuanhuang/kevislin/data/platform/Smart_seq2'
# query5_path = 'H:/yuanhuang/kevislin/data/platform/DropSeq'

query_paths = [
    'E:/yuanhuang/kevislin/data/species/EMTAB5061_GSE/human'
    # 'E:/yuanhuang/kevislin/data/platform/10X_V3/',
    # 'H:/yuanhuang/kevislin/data/platform/Seq_Well',
    # 'H:/yuanhuang/kevislin/data/platform/Smart_seq2',
    # 'H:/yuanhuang/kevislin/data/platform/DropSeq',
]
flags = {
    'PBMC': False,
    'E-MTAB': False,
    'E-MTAB query': False,
}

ref_label = pd.read_csv(os.path.join(ref_path, 'label.csv'))
query_labels = []
for path in query_paths:
    query_labels.append(pd.read_csv(os.path.join(path, 'label.csv')))
# query1_label = pd.read_csv(os.path.join(query1_path, 'label.csv'))


ref_label_key = 'type'
query_label_key = 'type'

ref_label.columns = [ref_label_key]
for label in query_labels:
    label.columns = [query_label_key]

# For e-mtab-5061
if flags['E-MTAB']:
    ref_label[ref_label_key] = ref_label[ref_label_key].apply(lambda x: x.split()[0])
if flags['E-MTAB query']:
    query_labels[0][query_label_key] = query_labels[0][query_label_key].apply(lambda x: x.split()[0])

    # for i in range(len(query_labels)):
    #     query_labels[i][query_label_key] = query_labels[i][query_label_key].apply(lambda x: x.split()[0])

ref_idx = (~ref_label[ref_label_key].isin(['Unassigned', 'unclear', 'not', 'co-expression', 'unclassified'])).tolist()
ref_label_list = ref_label.iloc[ref_idx][ref_label_key].tolist()
# 一对一情况 取交集(可选可不选)
ref_label_list = list(set(ref_label_list) & set(query_labels[0][query_label_key].tolist()))
query_indices = []
ref_idx2 = ref_label[ref_label_key].isin(ref_label_list).tolist()
ref_idx = [ref_idx[i] & ref_idx2[i] for i in range(len(ref_idx))]
# ref_label = ref_label.iloc[ref_idx, :]

for label in query_labels:
    query_indices.append(label[query_label_key].isin(ref_label_list).tolist())
# query1_idx = query1_label[query_label_key].isin(ref_label_list).tolist()

# print("raw query")
# print(set(query_labels[0][query_label_key].tolist()))
ref_label = ref_label.iloc[ref_idx, :]
for i in range(len(query_labels)):
    query_labels[i] = query_labels[i].iloc[query_indices[i], :]
# query1_label = query1_label.iloc[query1_idx, :]

print("ref label set :", sorted(list(set(ref_label[ref_label_key].tolist()))))

for i, label in enumerate(query_labels):
    print("query " + str(i + 1) + " label set: ", sorted(list(set(query_labels[i][query_label_key].tolist()))))
# print("query 1 label set :", sorted(list(set(query1_label[query_label_key].tolist()))))

ref_data = pd.read_csv(os.path.join(ref_path, 'data.csv'), index_col=0)
query_datas = []
for path in query_paths:
    query_datas.append(pd.read_csv(os.path.join(path, 'data.csv'), index_col=0))
# query1_data = pd.read_csv(os.path.join(query1_path, 'data.csv'), index_col=0)

# For PBMC1 dataset
if flags['PBMC']:
    ref_data.columns = [x.split("_")[1] for x in ref_data.columns.tolist()]
    for i in range(len(query_datas)):
        query_datas[i].columns = [x.split("_")[1] for x in query_datas[i].columns.tolist()]

    # query1_data.columns = [x.split("_")[1] for x in query1_data.columns.tolist()]

# 基因交集
# datasets = [query1_data, query2_data, query3_data, query4_data, query5_data]
# datasets = [query1_data]
gene_name = set(ref_data.columns.tolist())
for x in query_datas:
    gene_name = set(x.columns.tolist()) & gene_name

# 去掉表达量之和为0的基因
# 记得加上ref_data
datasets = []
datasets.append(ref_data)
datasets += query_datas

for x in datasets:
    df = x.sum(axis=0)
    df = df[df == 0]
    gene_to_del = set(df.index.tolist())
    gene_name -= gene_to_del
gene_name = list(gene_name)

ref_data = ref_data.iloc[ref_idx, :][gene_name]
for i in range(len(query_datas)):
    query_datas[i] = query_datas[i].iloc[query_indices[i], :][gene_name]
# query1_data = query1_data.iloc[query1_idx, :][gene_name]

print("ref data shape ", ref_data.shape)
for i, data in enumerate(query_datas):
    print("query " + str(i + 1) + " data shape", data.shape)

ref_data.to_csv(os.path.join(ref_save_path, 'data_1.csv'))
for i, data in enumerate(query_datas):
    data.to_csv(os.path.join(query_save_path, 'data_' + str(i + 1) + '.csv'))
# query1_data.to_csv(os.path.join(query_save_path, 'data_1.csv'))


ref_label.to_csv(os.path.join(ref_save_path, 'label_1.csv'), index=False)
for i, label in enumerate(query_labels):
    label.to_csv(os.path.join(query_save_path, 'label_' + str(i + 1) + '.csv'), index=False)

# query1_label.to_csv(os.path.join(query_save_path, 'label_1.csv'), index=False)
