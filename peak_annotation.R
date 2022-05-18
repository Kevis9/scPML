library(Seurat)
library(hdf5r)
library(ggplot2)
library(patchwork)

peaks <- Read10X_h5("atac_v1_pbmc_10k_filtered_peak_bc_matrix.h5")
# create a gene activity matrix from the peak matrix and GTF, using chromosomes 1:22, X, and Y.
# Peaks that fall within gene bodies, or 2kb upstream of a gene, are considered
activity.matrix <- CreateGeneActivityMatrix(peak.matrix = peaks, annotation.file = "Homo_sapiens.GRCh37.82.gtf", 
    seq.levels = c(1:22, "X", "Y"), upstream = 2000, verbose = TRUE)

pbmc.atac <- CreateSeuratObject(counts = peaks, assay = "ATAC", project = "10x_ATAC")
pbmc.atac[["ACTIVITY"]] <- CreateAssayObject(counts = activity.matrix)
meta <- read.table("atac_v1_pbmc_10k_singlecell.csv", sep = ",", header = TRUE, row.names = 1, 
    stringsAsFactors = FALSE)
meta <- meta[colnames(pbmc.atac), ]
pbmc.atac <- AddMetaData(pbmc.atac, metadata = meta)
pbmc.atac <- subset(pbmc.atac, subset = nCount_ATAC > 5000)
pbmc.atac$tech <- "atac"

DefaultAssay(pbmc.atac) <- "ACTIVITY"
pbmc.atac <- FindVariableFeatures(pbmc.atac)
pbmc.atac <- NormalizeData(pbmc.atac)
pbmc.atac <- ScaleData(pbmc.atac)

DefaultAssay(pbmc.atac) <- "ATAC"
VariableFeatures(pbmc.atac) <- names(which(Matrix::rowSums(pbmc.atac) > 100))
pbmc.atac <- RunLSI(pbmc.atac, n = 50, scale.max = NULL)
pbmc.atac <- RunUMAP(pbmc.atac, reduction = "lsi", dims = 1:50)

pbmc.rna <- readRDS("pbmc_10k_v3.rds")
pbmc.rna$tech <- "rna"

transfer.anchors <- FindTransferAnchors(reference = pbmc.rna, query = pbmc.atac, features = VariableFeatures(object = pbmc.rna), 
    reference.assay = "RNA", query.assay = "ACTIVITY", reduction = "cca")

celltype.predictions <- TransferData(anchorset = transfer.anchors, refdata = pbmc.rna$celltype, 
    weight.reduction = pbmc.atac[["lsi"]])
pbmc.atac <- AddMetaData(pbmc.atac, metadata = celltype.predictions)

pbmc.atac.filtered <- subset(pbmc.atac, subset = prediction.score.max > 0.5)
pbmc.atac.filtered$predicted.id <- factor(pbmc.atac.filtered$predicted.id, levels = levels(pbmc.rna))  # to make the colors match

DefaultAssay(pbmc.atac.filtered)  = 'ACTIVITY'

atac_data = GetAssayData(pbmc.atac.filtered)
atac_label = as.matrix(pbmc.atac.filtered$predicted.id)


# p1 <- DimPlot(pbmc.atac, reduction = "umap") + NoLegend() + ggtitle("scATAC-seq")

write.table(atac_data, file='atac_data.csv', sep=',', row.names=TRUE, col.names=TRUE,quote=FALSE)
write.table(atac_label, file='atac_label.csv', sep=',', row.names=TRUE, col.names=TRUE,quote=FALSE)