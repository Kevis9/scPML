library(Seurat)
library(hdf5r)
# install.packages("ggplot2", repos="https://mirrors.tuna.tsinghua.edu.cn/CRAN/")
# install.packages("patchwork", repos="https://mirrors.tuna.tsinghua.edu.cn/CRAN/")
# 加载peak矩阵文件


# ATAC activity matrix pre-process
# atac_data = t(read.csv('/home/zhianhuang/yuanhuang/kevislin/data/omics_data/A549/atac_data.csv',row.names=1)) # gene * cell
# print(rownames(atac_data))
# print(colnames(atac_data))
# atac_data = NormalizeData(atac_data)
# atac_data = ScaleData(atac_data)
#
# # RNA expression matrix pre-process
# rna_data = t(read.csv('/home/zhianhuang/yuanhuang/kevislin/data/omics_data/A549/rna_data.csv', row.names=1)) # gene * cell
# rna_data = NormalizeData(rna_data)
# rna_data = ScaleData(rna_data)




atac_data = t(read.csv('atac_middle_out.csv', row.names=1))
print(rownames(atac_data))
activity.matrix <- CreateGeneActivityMatrix(peak.matrix = atac_data,
 	annotation.file = "Homo_sapiens.GRCh37.82.gtf",
 	seq.levels = c(1:22, "X", "Y"),
 	upstream = 2000,
 	verbose = TRUE)


write.table(t(activity.matrix), file='atac_activity_mat.csv', sep=',', row.names=TRUE, col.names=TRUE,quote=FALSE);
# write.table(rna_data, file='rna_norm_data', sep=',', row.names=TRUE, col.names=TRUE,quote=FALSE);
