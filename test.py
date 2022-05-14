import scipy.io as spio
import pandas as pd


atac_data = spio.mmread('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/A549/ATAC/GSM3271041_ATAC_sciCAR_A549_peak_count.txt')
print(type(atac_data))
exit()

