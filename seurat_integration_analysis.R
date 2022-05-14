library(Seurat)
library(ggplot2)
library(patchwork)

atac_data = Matrix::readMM('/home/zhianhuang/yuanhuang/kevislin/data/raw_data/omics_data/A549/ATAC/GSM3271041_ATAC_sciCAR_A549_peak_count.txt')
atac_data = as.matrix(atac_data)

atac_chr = read.csv('/home/zhianhuang/yuanhuang/kevislin/data/raw_data/omics_data/A549/ATAC/GSM3271041_ATAC_sciCAR_A549_peak.txt')
atac_cell = read.csv('/home/zhianhuang/yuanhuang/kevislin/data/raw_data/omics_data/A549/ATAC/GSM3271041_ATAC_sciCAR_A549_cell.txt')

print(atac_chr['chr'])

# activity.matrix =