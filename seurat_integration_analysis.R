library(Seurat)
library(ggplot2)
library(patchwork)

# atac_data
atac_data = Matrix::readMM('atac_rna_to_R/atac_data.txt')
atac_data = t(as.matrix(atac_data))


atac_chr = read.csv('atac_rna_to_R/atac_chr.csv')
atac_cell = read.csv('atac_rna_to_R/atac_cell.csv')
print(atac_chr['peak'])
print(class(atac_chr['peak']))
q()
rownames(atac_data) = apply(atac_chr['peak'],1,as.list)
colnames(atac_data) = apply(atac_cell['sample'],1,as.list)


# rna_data
rna_data = Matrix::readMM('atac_rna_to_R/rna_data.txt')
rna_data = t(as.matrix(rna_data))

rna_gene = read.csv('atac_rna_to_R/rna_gene.csv')
rna_cell = read.csv('atac_rna_to_R/rna_cell.csv')


rownames(rna_data) = apply(rna_gene['gene_name'],1,as.list)
colnames(rna_data) = apply(rna_cell['sample'],1,as.list)



# 重头戏
print(rownames(atac_data))
print(atac_data)
activity.matrix <- CreateGeneActivityMatrix(peak.matrix = atac_data, annotation.file = "Homo_sapiens.GRCh37.82.gtf",
    seq.levels = c(1:22, "X", "Y"), upstream = 2000, verbose = TRUE)

pbmc.atac <- CreateSeuratObject(counts = peaks, assay = "ATAC", project = "10x_ATAC")
pbmc.atac[["ACTIVITY"]] <- CreateAssayObject(counts = activity.matrix)
pbmc.atac$tech <- "atac"


DefaultAssay(pbmc.atac) <- "ACTIVITY"
pbmc.atac <- FindVariableFeatures(pbmc.atac)
pbmc.atac <- NormalizeData(pbmc.atac)
pbmc.atac <- ScaleData(pbmc.atac)

pbmc.rna <- rna_data
pbmc.rna$tech <- "rna"


genes.use <- VariableFeatures(pbmc.rna)
refdata <- GetAssayData(pbmc.rna, assay = "RNA", slot = "data")[genes.use, ]

# refdata (input) contains a scRNA-seq expression matrix for the scRNA-seq cells.  imputation
# (output) will contain an imputed scRNA-seq matrix for each of the ATAC cells
imputation <- TransferData(anchorset = transfer.anchors, refdata = refdata, weight.reduction = pbmc.atac[["lsi"]])

# this line adds the imputed data matrix to the pbmc.atac object
pbmc.atac[["RNA"]] <- imputation
coembed <- merge(x = pbmc.rna, y = pbmc.atac)

# Finally, we run PCA and UMAP on this combined object, to visualize the co-embedding of both
# datasets
coembed <- ScaleData(coembed, features = genes.use, do.scale = FALSE)
coembed <- RunPCA(coembed, features = genes.use, verbose = FALSE)
coembed <- RunUMAP(coembed, dims = 1:30)


p1 <- DimPlot(coembed, group.by = "tech")

ggsave('atac_rna_to_R/atac_rna.png', p1)


# activity.matrix =