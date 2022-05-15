import pandas as pd
import scipy.io as spio
from scipy import sparse
atac_data = spio.mmread('/home/zhianhuang/yuanhuang/kevislin/data/raw_data/omics_data/A549/ATAC/GSM3271041_ATAC_sciCAR_A549_peak_count.txt')
atac_data = atac_data.todense().T


atac_cell = pd.read_csv('/home/zhianhuang/yuanhuang/kevislin/data/raw_data/omics_data/A549/ATAC/GSM3271041_ATAC_sciCAR_A549_cell.txt')
atac_chr = pd.read_csv('/home/zhianhuang/yuanhuang/kevislin/data/raw_data/omics_data/A549/ATAC/GSM3271041_ATAC_sciCAR_A549_peak.txt', dtype={'chr':object})

atac_df = pd.DataFrame(data=atac_data)

# 取A549 Cell
cell_idx = atac_cell['group'].str.startswith('A549').tolist()

atac_cell = atac_cell.iloc[cell_idx, :]
atac_df = atac_df.iloc[cell_idx, :]

# 保留1~22, X, Y染色体
# chr_arr = [str(x) for x in range(1, 23)]
#
# chr_arr += ['X', 'Y']
# chr_idx = atac_chr['chr'].isin(chr_arr).tolist()
#
#
# atac_chr = atac_chr.iloc[chr_idx, :]
#
# atac_df = atac_df.iloc[:, chr_idx]

# 设置index和columns, 注意columns的格式
atac_df.index = atac_cell['sample'].tolist()
atac_df.columns = (atac_chr['peak'].map(lambda x: ('chr' + x).replace('-', ':', 1))).tolist()


def save_to_mm(df, str1, str2):
    # 然后转成稀疏矩阵，进行存储
    colnames = pd.DataFrame(data=df.columns.tolist(), columns=[str1])
    rownames = pd.DataFrame(data=df.index.tolist(), columns=[str2])
    data = df.to_numpy()
    data = sparse.coo_matrix(df)
    return rownames, colnames, data


atac_cell, atac_chr, atac_data = save_to_mm(atac_df, 'peak', 'sample')


atac_chr.to_csv('atac_rna_to_R/atac_chr.csv', index=False)
atac_cell.to_csv('atac_rna_to_R/atac_cell.csv', index=False)
spio.mmwrite('atac_rna_to_R/atac_data.txt', atac_data)



# RNA
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

rna_df = rna_df.T
rna_df = rna_df.groupby(rna_df.index).sum()
rna_df = rna_df.T

cell_df, gene_df, gene_data = save_to_mm(rna_df, 'gene_name', 'sample')

gene_df.to_csv('atac_rna_to_R/rna_gene.csv', index=False)
cell_df.to_csv('atac_rna_to_R/rna_cell.csv', index=False)
spio.mmwrite('atac_rna_to_R/gene_data.txt', gene_data)


