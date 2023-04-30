import os.path

import pandas as pd

raw_ref_data = pd.read_csv(r'E:\YuAnHuang\kevislin\data\unknown_cell_type\GSE72056\raw_data.txt', index_col=0, sep='\t')
raw_query_data = pd.read_csv(r'E:\YuAnHuang\kevislin\data\unknown_cell_type\GSE103322\raw_data.txt', index_col=0, sep='\t')


raw_ref_data = raw_ref_data.T
raw_query_data = raw_query_data.T

# non-malignant cell type (1=T,2=B,3=Macro.4=Endo.,5=CAF;6=NK)
del raw_ref_data['tumor']
idx = raw_ref_data['malignant(1=no,2=yes,0=unresolved)'].isin([2]).tolist()

raw_ref_data.loc[idx, 'non-malignant cell type (1=T,2=B,3=Macro.4=Endo.,5=CAF;6=NK)'] = 'malignant'
map_dic = {1: "T cell",
           2: "B cell",
           3: "Macrophage",
           4: "Endothelial",
           5: "CAF",
           6: "NK"}

raw_ref_data['non-malignant cell type (1=T,2=B,3=Macro.4=Endo.,5=CAF;6=NK)'] = raw_ref_data['non-malignant cell type (1=T,2=B,3=Macro.4=Endo.,5=CAF;6=NK)'].replace(map_dic)

idx = (~raw_ref_data['non-malignant cell type (1=T,2=B,3=Macro.4=Endo.,5=CAF;6=NK)'].isin([0])).tolist()
raw_ref_data = raw_ref_data.iloc[idx, :]

raw_ref_label = pd.DataFrame(data = raw_ref_data['non-malignant cell type (1=T,2=B,3=Macro.4=Endo.,5=CAF;6=NK)'].tolist(), columns=['type'])
del raw_ref_data['non-malignant cell type (1=T,2=B,3=Macro.4=Endo.,5=CAF;6=NK)']
del raw_ref_data['malignant(1=no,2=yes,0=unresolved)']

idx = raw_query_data['classified  as cancer cell'] == 1
idx= idx.tolist()
raw_query_data.loc[idx, 'non-cancer cell type'] = 'malignant'
raw_query_data['non-cancer cell type'] = raw_query_data['non-cancer cell type'].replace({0: 'malignant'})

idx = (~raw_query_data['non-cancer cell type'].isin([0])).tolist()
raw_query_data = raw_query_data.iloc[idx, :]

raw_query_label = pd.DataFrame(data=raw_query_data['non-cancer cell type'].tolist(), columns=['type'])

del raw_query_data['non-cancer cell type']
del raw_query_data['classified  as cancer cell']
del raw_query_data['classified as non-cancer cells']
del raw_query_data['processed by Maxima enzyme']
del raw_query_data['Lymph node']


# 取ref 和 query的细胞交集

# 先对query的genes做一定的处理
raw_query_data.columns = raw_query_data.columns.map(lambda x: x.split("'")[1])
common_type = list(set(raw_ref_label['type'].tolist()) & set(raw_query_label['type'].tolist()))

# 去掉这个Endothelial试试
common_type.remove('Endothelial')


ref_idx = raw_ref_label['type'].isin(common_type).tolist()
query_idx = raw_query_label['type'].isin(common_type).tolist()
ref_data = raw_ref_data.iloc[ref_idx,:]
query_data = raw_query_data.iloc[query_idx,:]
ref_label = raw_ref_label.iloc[ref_idx, :]
query_label = raw_query_label.iloc[query_idx, :]



# 取基因的交集，先对refdata和query data重复的基因做一个处理
# 对于重复基因，参考scGCN，取基因表达量最大的那个
def drop_duplicate_columns(data):
    counter = {}
    for i, gene in enumerate(data.columns.tolist()):
        if gene not in counter:
            counter[gene] = [i]
        else:
            counter[gene].append(i)
    col_idx = []
    for gene, indices in counter.items():
        if len(indices) > 1:
            # 重复的基因
            max_idx = 0
            max_num = -1
            for i in indices:
                if max_num < data.iloc[:, i].sum():
                    max_idx = i
                    max_num = data.iloc[:, i].sum()
            col_idx.append(max_idx)
        else:
            col_idx.append(indices[0])
    # 尽量保持原来的顺序不变（个人习惯）
    col_idx.sort()
    data = data.iloc[:, col_idx]
    return data


ref_data = drop_duplicate_columns(ref_data)
query_data =  drop_duplicate_columns(query_data)
common_gene = list(set(raw_ref_data.columns.tolist()) & set(raw_query_data.columns.tolist()))
ref_data = ref_data[common_gene]
query_data = query_data[common_gene]
print(ref_data.shape)
print(query_data.shape)

# 交换一下
tmp = ref_data
ref_data = query_data
query_data = tmp

tmp = ref_label
ref_label = query_label
query_label = tmp

# ref_data.to_csv('ref_data.csv')
# query_data.to_csv('query_data.csv')
# ref_label.to_csv('ref_label.csv', index=False)
# query_label.to_csv('query_label.csv', index=False)

# 去掉ref每种细胞的类型
# 然后保存ref
for label in common_type:
    if not label == 'malignant':
        continue
    ref_idx = (~(ref_label['type']==label)).tolist()
    # path = 'GSE72056_GSE103322_' + label
    path = 'GSE103322_GSE72056' + label
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

    # path = 'GSE72056_GSE103322_' + label
    path = 'GSE103322_GSE72056' + label
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