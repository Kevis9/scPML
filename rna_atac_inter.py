'''
    对ATAC-seq的activity mat和RNA的expression mat取交集:
    1. 对RNA mat的处理: 取 A549 cell
    2. 两者的第一行（基因名称）取交集
    3. 取两者的交集
    4. 获取到最后的label
'''

import scipy.io as spio
import pandas as pd

rna_data = spio.mmread('/home/zhianhuang/yuanhuang/kevislin/data/raw_data/omics_data/A549/RNA/GSM3271040_RNA_sciCAR_A549_gene_count.txt')
rna_data = rna_data.todense().T


rna_cell = pd.read_csv('/home/zhianhuang/yuanhuang/kevislin/data/raw_data/omics_data/A549/RNA/GSM3271040_RNA_sciCAR_A549_cell.txt')
rna_gene = pd.read_csv('/home/zhianhuang/yuanhuang/kevislin/data/raw_data/omics_data/A549/RNA/GSM3271040_RNA_sciCAR_A549_gene.txt')

rna_df = pd.DataFrame(data=rna_data)
# 取A549 cell
cell_idx = (rna_cell['cell_name'] == 'A549').tolist()
rna_cell = rna_cell.iloc[cell_idx, :]
rna_df = rna_df.iloc[cell_idx, :]

# 设置rna_df的index和columns
rna_df.index = rna_cell['sample'].tolist()
rna_df.columns = rna_gene['gene_short_name'].tolist()



# 细胞的交集,atac这里读入是 gene * cell, 所以要做一个转置
atac_activity_df = pd.read_csv('atac_activity_mat.csv', index_col=0).T
atac_cell_name = atac_activity_df.index.str.replace(".", "-", 3).tolist()
atac_activity_df.index = atac_cell_name
common_cell = list(set(atac_cell_name) & set(rna_df.index.tolist()))

atac_df = atac_activity_df.loc[common_cell, :]
rna_df = rna_df.loc[common_cell, :]

print(rna_df.shape)
print(atac_df.shape)

# label
label_idx = (rna_cell['sample'].isin(rna_df.index.tolist())).tolist()
label_df = rna_cell.iloc[label_idx, :]['treatment_time']


# 对RNA做gene selection
# Create and fit selector

# selector = SelectKBest(f_classif, k=10000)
# selector.fit(rna_df.to_numpy(), label_df.to_numpy())
# cols = selector.get_support(indices=True)
# rna_df = rna_df.iloc[:, cols]
# print(rna_df.shape)

# label_df.to_csv('label.csv', index=False)
# data = reduce_dimension(rna_df)
# show_cluster(data, label_df.to_numpy().reshape(-1), 'rna_8000_v2')
# exit()
# selector.fit(atac_df.to_numpy(), label_df.to_numpy())
# cols = selector.get_support(indices=True)
# atac_df = atac_df.iloc[:, cols]

# 基因的交集
commom_gene = list(set(atac_df.columns.tolist()) & set(rna_df.columns.tolist()))

atac_df = atac_df.loc[:, commom_gene]
rna_df = rna_df.loc[:, commom_gene]

# 将rna中重复的gene加起来
rna_df = rna_df.T
rna_df = rna_df.groupby(rna_df.index).sum()
rna_df = rna_df.T

# 再次将gene顺序调整好
rna_df = rna_df[atac_df.columns.tolist()]


# 保存
print(rna_df.shape)
print(atac_df.shape)
print(label_df.shape)

rna_df.to_csv('data_1.csv')
atac_df.to_csv('atac_data.csv')
label_df.to_csv('label.csv', index=False)




