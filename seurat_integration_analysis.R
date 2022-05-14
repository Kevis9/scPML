library(Seurat)
library(ggplot2)
library(patchwork)

peaks = Matrix::readMM('/home/zhianhuang/yuanhuang/kevislin/data/raw_data/omics_data/A549/ATAC/GSM3271041_ATAC_sciCAR_A549_peak_count.txt')
print(as.matrix(peaks))
