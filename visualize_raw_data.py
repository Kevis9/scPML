# import numpy as np
# import pandas as pd
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import f_classif
#
#
#
# import scipy.io as spio
# rna_data = spio.mmread(
#         '/home/zhianhuang/yuanhuang/kevislin/data/raw_data/omics_data/A549/RNA/GSM3271040_RNA_sciCAR_A549_gene_count.txt')
# rna_data = rna_data.todense().T
# rna_gene = pd.read_csv(
#         '/home/zhianhuang/yuanhuang/kevislin/data/raw_data/omics_data/A549/RNA/GSM3271040_RNA_sciCAR_A549_gene.txt')
# rna_cell = pd.read_csv('/home/zhianhuang/yuanhuang/kevislin/data/raw_data/omics_data/A549/RNA/GSM3271040_RNA_sciCAR_A549_cell.txt')
# rna_df = pd.DataFrame(data=rna_data, index=rna_cell['sample'].tolist(), columns=rna_gene['gene_short_name'])
# idx = (rna_cell['cell_name']=='A549').tolist()
#
# rna_df = rna_df.iloc[idx, :]
# label_df = rna_cell.iloc[idx, :]['treatment_time']
#
#
# # atac_df = pd.read_csv('/Users/kevislin/Desktop/a549_data/atac_data.csv', index_col=0)
# # label_df = pd.read_csv('/Users/kevislin/Desktop/a549_data/label.csv')
#
# # quality control
# # sum_row = rna_df.apply(lambda x: x.sum(), axis=1)
# # q_idx = (sum_row > 500).tolist()
# # rna_df = rna_df.iloc[q_idx, :]
# # label_df = label_df.iloc[q_idx, :]
#
# # print(rna_df.shape)
#
# # rna_inter_df = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/omics_data/A549/rna_data.csv', index_col=0)
# #
# # gene_names = rna_inter_df.columns.tolist()
# #
# # rna_df = rna_df[gene_names]
#
#
#
#
#
#
# # atac_data = atac_df.to_numpy()
# # atac_data = sc.pp.normalize_total(atac_data)
# # atac_data = sc.pp.scale(atac_data)
#
#
# selector = SelectKBest(f_classif, k=15000)
# selector.fit(rna_df.to_numpy(), label_df.to_numpy())
# cols = selector.get_support(indices=True)
# rna_df = rna_df.iloc[:, cols]
#
#
#
# data_2d = reduce_dimension(rna_df.to_numpy())
# # label = label.reshape(-1)
#
# # label = ['rna' for i in range(rna_data.shape[0])]
# # label += ['atac' for i in range(rna_data.shape[0])]
# show_cluster(data_2d, label_df.to_numpy(), 'rna_15000')


