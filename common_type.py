import os
import pandas as pd
from functools import reduce
'''
1. 细胞类型取交集， 可以左连接(可选)
2. 基因取交集，去掉表达为0的基因(可选)
'''
save_path = 'experiment/multi_ref/PBMC2/raw_data'

# raw_data_save_path= 'e/xperiment/platform/new_version/seq_well_drop_seq/raw_data'

path = 'E:/YuAnHuang/kevislin/data/platform/10X_V2'
ref_paths = [
    'E:/YuAnHuang/kevislin/data/platform/Seq_Well',
    'E:/YuAnHuang/kevislin/data/platform/DropSeq',
    'E:/YuAnHuang/kevislin/data/platform/10X_V3'
]

query_paths = [
    'E:/YuAnHuang/kevislin/data/platform/InDrop'
    # 'E:/yuanhuang/kevislin/data/platform/GSE85241',
    # 'E:/yuanhuang/kevislin/data/platform/10X_V3/',
    # 'H:/yuanhuang/kevislin/data/platform/Seq_Well',
]
flags = {
    'PBMC query': True,
    'PBMC ref' : True,
    'E-MTAB': False, #旧的EM用
    'E-MTAB query': False,
    'get_rid_of_gene':False, #去掉表达量为0的基因
    'if ref has 2000 genes':False,
}

ref_save_path = os.path.join(save_path, 'ref')
query_save_path = os.path.join(save_path, 'query')

# ref 和 query 的label读取
ref_labels = []
for path in ref_paths:
    ref_labels.append(pd.read_csv(os.path.join(path, 'label.csv')))
query_labels = []
for path in query_paths:
    query_labels.append(pd.read_csv(os.path.join(path, 'label.csv')))
# query1_label = pd.read_csv(os.path.join(query1_path, 'label.csv'))


# 对ref和query label设置好统一的列名
ref_label_key = 'type'
query_label_key = 'type'
for label in  ref_labels:
    label.columns = [ref_label_key]
# ref_label.columns = [ref_label_key]
for label in query_labels:
    label.columns = [query_label_key]

# For e-mtab-5061： 这里还需要自己设置一下ref和query在第几个数据是emtab
if flags['E-MTAB']:
    ref_labels[0][ref_label_key] = ref_labels[0][ref_label_key].apply(lambda x: x.split()[0])
if flags['E-MTAB query']:
    query_labels[0][query_label_key] = query_labels[0][query_label_key].apply(lambda x: x.split()[0])


# 去掉一些不必要的细胞类型 以及 取细胞类型的交集
def intersect(a, b):
    return a & b
ref_common_cell = []
for label in ref_labels:
    ref_common_cell.append(set(label[ref_label_key].tolist()))
ref_common_cell = reduce(intersect, ref_common_cell)
ref_common_cell = ref_common_cell - (ref_common_cell & set(['Unassigned', 'unclear', 'not', 'co-expression', 'unclassified']))


ref_indices = []
for label in ref_labels:
    ref_indices.append((label[ref_label_key].isin(ref_common_cell)).tolist())
# ref_idx = (~ref_label[ref_label_key].isin(['Unassigned', 'unclear', 'not', 'co-expression', 'unclassified'])).tolist()
ref_label_list = list(ref_common_cell)

# ref_label_list = ref_label.iloc[ref_idx][ref_label_key].tolist()
# 一对一情况 取交集(可选可不选)
# ref_label_list = list(set(ref_label_list) & set(query_labels[0][query_label_key].tolist()))

query_indices = []

# ref_idx2 = ref_label[ref_label_key].isin(ref_label_list).tolist()
# ref_idx = [ref_idx[i] & ref_idx2[i] for i in range(len(ref_idx))]
# ref_label = ref_label.iloc[ref_idx, :]

for label in query_labels:
    query_indices.append(label[query_label_key].isin(ref_label_list).tolist())
# query1_idx = query1_label[query_label_key].isin(ref_label_list).tolist()

# print("raw query")
# print(set(query_labels[0][query_label_key].tolist()))

# ref_label = ref_label.iloc[ref_idx, :]
for i in range(len(ref_labels)):
    ref_labels[i] = ref_labels[i].iloc[ref_indices[i], :]

for i in range(len(query_labels)):
    query_labels[i] = query_labels[i].iloc[query_indices[i], :]
# query1_label = query1_label.iloc[query1_idx, :]

print("ref label set :", sorted(ref_label_list))

for i, label in enumerate(query_labels):
    print("query " + str(i + 1) + " label set: ", sorted(list(set(query_labels[i][query_label_key].tolist()))))

# print("query 1 label set :", sorted(list(set(query1_label[query_label_key].tolist()))))
ref_datas = []
for path in ref_paths:
    ref_datas.append(pd.read_csv(os.path.join(path, 'data.csv'), index_col=0))
# ref_data = pd.read_csv(os.path.join(ref_path, 'data.csv'), index_col=0)
# print("reference data shape {:}".format(ref_data.shape))
query_datas = []
for path in query_paths:
    query_datas.append(pd.read_csv(os.path.join(path, 'data.csv'), index_col=0))
# query1_data = pd.read_csv(os.path.join(query1_path, 'data.csv'), index_col=0)


# For PBMC1 dataset
# 在基因名字上要做一定的处理
if flags['PBMC query']:
    for i in range(len(query_datas)):
        query_datas[i].columns = [x.split("_")[1] for x in query_datas[i].columns.tolist()]

if flags['PBMC ref']:
    for i in range(len(ref_datas)):
        ref_datas[i].columns = [x.split("_")[1] for x in ref_datas[i].columns.tolist()]
    # ref_data.columns = [x.split("_")[1] for x in ref_data.columns.tolist()]

    # query1_data.columns = [x.split("_")[1] for x in query1_data.columns.tolist()]

# 如果reference 是经过preprocess.R处理之后的数据(原来的gene中的-符号全部变成.符号),那么对query做同样的处理
if flags['if ref has 2000 genes']:
    for i in range(len(query_datas)):
        query_datas[i].columns = query_datas[i].columns.map(lambda x: x.replace("-", ".")).tolist()


# 基因交集
gene_name = []
for data in ref_datas + query_datas :
    gene_name.append(set(data.columns.tolist()))
# gene_name = set(ref_data.columns.tolist())
# for x in query_datas:
#     gene_name = set(x.columns.tolist()) & gene_name
gene_name = reduce(intersect, gene_name)


# 去掉表达量之和为0的基因
# 记得加上ref_data
if flags['get_rid_of_gene']:
    datasets = []
    datasets += ref_datas
    datasets += query_datas
    for x in datasets:
        df = x.sum(axis=0)
        df = df[df == 0]
        gene_to_del = set(df.index.tolist())
        gene_name -= gene_to_del

gene_name = list(gene_name)
# ref_data = ref_data.iloc[ref_idx, :][gene_name]
for i in range(len(ref_datas)):
    ref_datas[i] = ref_datas[i].iloc[ref_indices[i], :][gene_name]

for i in range(len(query_datas)):
    query_datas[i] = query_datas[i].iloc[query_indices[i], :][gene_name]
# query1_data = query1_data.iloc[query1_idx, :][gene_name]

# print("ref data shape ", ref_data.shape)
for i, data in enumerate(ref_datas):
    print("ref " + str(i + 1) + " data shape", data.shape)

for i, data in enumerate(query_datas):
    print("query " + str(i + 1) + " data shape", data.shape)


'''
    save data
'''

# ref_data.to_csv(os.path.join(ref_save_path, 'data_1.csv'))
for i, data in enumerate(ref_datas):
    data.to_csv(os.path.join(ref_save_path, 'data_' + str(i + 1) + '.csv'))

for i, data in enumerate(query_datas):
    data.to_csv(os.path.join(query_save_path, 'data_' + str(i + 1) + '.csv'))

# label
# ref_label.to_csv(os.path.join(ref_save_path, 'label_1.csv'), index=False)
for i, label in enumerate(ref_labels):
    label.to_csv(os.path.join(ref_save_path, 'label_' + str(i + 1) + '.csv'), index=False)

for i, label in enumerate(query_labels):
    label.to_csv(os.path.join(query_save_path, 'label_' + str(i + 1) + '.csv'), index=False)