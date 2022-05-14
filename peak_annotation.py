'''
   Python处理得到ATAC的中间数据:
   1. 取A549 cell
   2. 只保留1~22 , X, Y染色体, 并且更改染色体的名称的样式 : chrX:12-123
   3. 保存这样的矩阵 ( chr * cell )
'''
import scipy.io as spio
import pandas as pd

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
chr_arr = [str(x) for x in range(1, 23)]

chr_arr += ['X', 'Y']
chr_idx = atac_chr['chr'].isin(chr_arr).tolist()


atac_chr = atac_chr.iloc[chr_idx, :]

atac_df = atac_df.iloc[:, chr_idx]

# 设置index和columns, 注意columns的格式
atac_df.index = atac_cell['sample'].tolist()
atac_df.columns = (atac_chr['peak'].map(lambda x: ('chr' + x).replace('-', ':', 1))).tolist()


# 保存
(atac_df.T).to_csv('atac_middle_out.csv')