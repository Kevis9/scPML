from calendar import c
from operator import index
import pandas as pd

rna_label = pd.read_csv('rna_label.csv')
atac_label = pd.read_csv('atac_label.csv')

c_type = list(set(rna_label['V1'].tolist()) & set(atac_label['V1'].tolist()))

rna_idx = rna_label['V1'].isin(c_type)
atac_idx = atac_label['V1'].isin(c_type)

rna_df = pd.read_csv('rna_data.csv', index_col=0)
atac_df = pd.read_csv('atac_data.csv', index_col=0)

rna_df = rna_df.iloc[rna_idx, :]
atac_df = atac_df.iloc[atac_idx, :]
rna_label = rna_label.iloc[rna_idx, :]
atac_label = atac_label.iloc[atac_idx, :]

print(rna_df.shape)
print(rna_label.shape)
print(atac_df.shape)
print(atac_label.shape)

rna_df.to_csv('rna_data2.csv')
atac_df.to_csv('atac_data2.csv')
rna_label.to_csv('rna_label2.csv', index=False)
atac_label.to_csv('atac_label2.csv', index=False)
