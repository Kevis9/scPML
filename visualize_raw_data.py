import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from utils import reduce_dimension, show_cluster
import scanpy as sc

# rna_data = pd.read_csv('/home/zhianhuang/yuanhuang/kevislin/data/omics_data/A549/rna_data.csv')
# label = pd.read_csv('/home/zhianhuang/yuanhuang/kevislin/data/omics_data/A549/rna_label.csv')
# # Quality control
# sum_row = rna_data.apply(lambda x: x.sum(), axis=1)
# idx = (sum_row > 5000).tolist()
# rna_data = rna_data.iloc[idx, :]


import scipy.io as spio
# rna_data = spio.mmread(
#         '/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/A549/RNA/GSM3271040_RNA_sciCAR_A549_gene_count.txt')
# rna_data = rna_data.todense().T
# rna_gene = pd.read_csv(
#         '/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/A549/RNA/GSM3271040_RNA_sciCAR_A549_gene.txt')
# rna_cell = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/A549/RNA/GSM3271040_RNA_sciCAR_A549_cell.txt')
# rna_df = pd.DataFrame(data=rna_data, index=rna_cell['sample'].tolist(), columns=rna_gene['gene_short_name'])
# idx = (rna_cell['cell_name']=='A549').tolist()
#
# rna_df = rna_df.iloc[idx, :]
# label_df = rna_cell.iloc[idx, :]

rna_df = pd.read_csv('/Users/kevislin/Desktop/a549_data/rna_data.csv', index_col=0)
atac_df = pd.read_csv('/Users/kevislin/Desktop/a549_data/atac_data.csv', index_col=0)
# label_df = pd.read_csv('/Users/kevislin/Desktop/a549_data/label.csv')

# quality control
# sum_row = rna_df.apply(lambda x: x.sum(), axis=1)
# q_idx = (sum_row > 500).tolist()
# rna_df = rna_df.iloc[q_idx, :]
# label_df = label_df.iloc[q_idx, :]

# print(rna_df.shape)

# rna_inter_df = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/omics_data/A549/rna_data.csv', index_col=0)
#
# gene_names = rna_inter_df.columns.tolist()
#
# rna_df = rna_df[gene_names]



rna_data = rna_df.to_numpy()
rna_data = sc.pp.normalize_total(rna_data)
rna_data = sc.pp.scale(rna_data)

atac_data = atac_df.to_numpy()
atac_data = sc.pp.normalize_total(atac_data)
atac_data = sc.pp.scale(atac_data)

# label = label_df.to_numpy()

# rna_data = SelectKBest(f_classif, k=10000).fit_transform(rna_df, label)

data_2d = reduce_dimension(np.concatenate([rna_data, atac_data], axis=0))


# label = label.reshape(-1)

label = ['rna' for i in range(rna_data.shape[0])]
label += ['atac' for i in range(rna_data.shape[0])]
show_cluster(data_2d, label, 'rna-atac')


