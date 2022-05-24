library(Seurat)
library(hdf5r)
library(ggplot2)
library(patchwork)

# peaks <- Read10X_h5("atac_v1_pbmc_10k_filtered_peak_bc_matrix.h5")
# create a gene activity matrix from the peak matrix and GTF, using chromosomes 1:22, X, and Y.
# chromosomes = c('chr1', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
#                 'chr2', 'chr20', 'chr21', 'chr22', 'chr3', 'chr4', 'chr5', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9',
#                 'GL000191.1', 'GL000192.1', 'GL000193.1', 'GL000194.1', 'GL000195.1', 'GL000196.1', 'GL000198.1',
#                 'GL000199.1', 'GL000201.1', 'GL000202.1', 'GL000204.1', 'GL000205.1', 'GL000208.1', 'GL000209.1',
#                 'GL000211.1', 'GL000212.1', 'GL000213.1', 'GL000214.1', 'GL000215.1', 'GL000216.1', 'GL000217.1',
#                 'GL000218.1', 'GL000219.1', 'GL000220.1', 'GL000221.1', 'GL000222.1', 'GL000223.1', 'GL000224.1',
#                 'GL000225.1', 'GL000226.1', 'GL000227.1', 'GL000228.1', 'GL000229.1', 'GL000230.1', 'GL000231.1',
#                 'GL000232.1', 'GL000233.1', 'GL000234.1', 'GL000236.1', 'GL000237.1', 'GL000240.1', 'GL000241.1',
#                 'GL000242.1', 'GL000243.1', 'GL000247.1', 'GL000250.1', 'GL000251.1', 'GL000252.1', 'GL000253.1',
#                 'GL000254.1', 'GL000255.1', 'GL000256.1', 'GL000257.1', 'GL000258.1', 'GL339449.2', 'GL339450.1',
#                 'GL383516.1', 'GL383517.1', 'GL383518.1', 'GL383519.1', 'GL383520.1', 'GL383521.1', 'GL383522.1',
#                 'GL383523.1', 'GL383526.1', 'GL383527.1', 'GL383528.1', 'GL383531.1', 'GL383533.1', 'GL383534.2',
#                 'GL383535.1', 'GL383536.1', 'GL383537.1', 'GL383538.1', 'GL383539.1', 'GL383540.1', 'GL383541.1',
#                 'GL383542.1', 'GL383543.1', 'GL383545.1', 'GL383546.1', 'GL383548.1', 'GL383550.1', 'GL383551.1',
#                 'GL383552.1', 'GL383553.2', 'GL383554.1', 'GL383555.1', 'GL383556.1', 'GL383557.1', 'GL383558.1',
#                 'GL383559.2', 'GL383560.1', 'GL383561.2', 'GL383562.1', 'GL383563.2', 'GL383564.1', 'GL383565.1',
#                 'GL383566.1', 'GL383567.1', 'GL383569.1', 'GL383571.1', 'GL383572.1', 'GL383573.1', 'GL383574.1',
#                 'GL383575.2', 'GL383576.1', 'GL383577.1', 'GL383579.1', 'GL383580.1', 'GL383581.1', 'GL383582.2',
#                 'GL383583.1', 'GL582967.1', 'GL582968.1', 'GL582969.1', 'GL582970.1', 'GL582971.1', 'GL582972.1',
#                 'GL582973.1', 'GL582974.1', 'GL582975.1', 'GL582976.1', 'GL582977.2', 'GL582979.2', 'GL877870.2',
#                 'GL877871.1', 'GL877872.1', 'GL877873.1', 'GL877875.1', 'GL877876.1', 'GL877877.2', 'GL949741.1',
#                 'GL949742.1', 'GL949743.1', 'GL949744.1', 'GL949745.1', 'GL949746.1', 'GL949747.1', 'GL949748.1',
#                 'GL949749.1', 'GL949750.1', 'GL949751.1', 'GL949752.1', 'GL949753.1', 'JH159131.1', 'JH159132.1',
#                 'JH159133.1', 'JH159134.2', 'JH159135.2', 'JH159136.1', 'JH159137.1', 'JH159138.1', 'JH159139.1',
#                 'JH159140.1', 'JH159141.2', 'JH159142.2', 'JH159143.1', 'JH159144.1', 'JH159145.1', 'JH159146.1',
#                 'JH159147.1', 'JH159148.1', 'JH159149.1', 'JH159150.3', 'JH591181.2', 'JH591182.1', 'JH591183.1',
#                 'JH591184.1', 'JH591185.1', 'JH591186.1', 'JH636052.4', 'JH636053.3', 'JH636054.1', 'JH636055.1',
#                 'JH636056.1', 'JH636057.1', 'JH636058.1', 'JH636059.1', 'JH636060.1', 'JH636061.1', 'JH720443.2',
#                 'JH720444.2', 'JH720445.1', 'JH720446.1', 'JH720447.1', 'JH720449.1', 'JH720451.1', 'JH720452.1',
#                 'JH720453.1', 'JH720454.3', 'JH720455.1', 'JH806576.1', 'JH806577.1', 'JH806578.1', 'JH806579.1',
#                 'JH806580.1', 'JH806581.1', 'JH806582.2', 'JH806583.1', 'JH806584.1', 'JH806585.1', 'JH806586.1',
#                 'JH806587.1', 'JH806588.1', 'JH806589.1', 'JH806590.2', 'JH806591.1', 'JH806592.1', 'JH806593.1',
#                 'JH806594.1', 'JH806595.1', 'JH806596.1', 'JH806597.1', 'JH806598.1', 'JH806599.1', 'JH806600.2',
#                 'JH806601.1', 'JH806602.1', 'JH806603.1', 'KB021645.1', 'KB021646.2', 'KB021647.1', 'KB021648.1',
#                 'KB663603.1', 'KB663604.1', 'KB663605.1', 'KB663606.1', 'KB663607.2', 'KB663608.1', 'KB663609.1',
#                 'KE332495.1', 'KE332496.1', 'KE332497.1', 'KE332498.1', 'KE332499.1', 'KE332501.1', 'KE332502.1',
#                 'KE332505.1', 'KE332506.1', 'M', 'chrX', 'chrY', 'hs37d5')

chromosomes = c('chr1', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
                'chr2', 'chr20', 'chr21', 'chr22', 'chr3', 'chr4', 'chr5', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chrX', 'chrY')

atac_data = read.csv('atac_middle_out.csv', row.names=1)
# print(rownames(atac_data))
# activity.matrix <- CreateGeneActivityMatrix(peak.matrix = atac_data,
#  	annotation.file = "Homo_sapiens.GRCh37.82.gtf",
#  	seq.levels = c(1:22, "X", "Y"),
#  	upstream = 2000,
#  	verbose = TRUE)

activity.matrix <- CreateGeneActivityMatrix(peak.matrix = atac_data,
 	annotation.file = "gencode.v40.chr_patch_hapl_scaff.annotation.gtf",
 	seq.levels = chromosomes,
 	upstream = 2000,
 	verbose = TRUE)


write.table(activity.matrix, file='atac_activity_mat.csv', sep=',', row.names=TRUE, col.names=TRUE,quote=FALSE)


# Peaks that fall within gene bodies, or 2kb upstream of a gene, are considered
# activity.matrix <- CreateGeneActivityMatrix(peak.matrix = peaks, annotation.file = "Homo_sapiens.GRCh37.82.gtf",
#     seq.levels = c(1:22, "X", "Y"), upstream = 2000, verbose = TRUE)
#
# pbmc.atac <- CreateSeuratObject(counts = peaks, assay = "ATAC", project = "10x_ATAC")
# pbmc.atac[["ACTIVITY"]] <- CreateAssayObject(counts = activity.matrix)
# meta <- read.table("atac_v1_pbmc_10k_singlecell.csv", sep = ",", header = TRUE, row.names = 1,
#     stringsAsFactors = FALSE)
# meta <- meta[colnames(pbmc.atac), ]
# pbmc.atac <- AddMetaData(pbmc.atac, metadata = meta)
# pbmc.atac <- subset(pbmc.atac, subset = nCount_ATAC > 5000)
# pbmc.atac$tech <- "atac"
#
# DefaultAssay(pbmc.atac) <- "ACTIVITY"
# pbmc.atac <- FindVariableFeatures(pbmc.atac)
# pbmc.atac <- NormalizeData(pbmc.atac)
# pbmc.atac <- ScaleData(pbmc.atac)
#
# DefaultAssay(pbmc.atac) <- "ATAC"
# VariableFeatures(pbmc.atac) <- names(which(Matrix::rowSums(pbmc.atac) > 100))
# pbmc.atac <- RunLSI(pbmc.atac, n = 50, scale.max = NULL)
# pbmc.atac <- RunUMAP(pbmc.atac, reduction = "lsi", dims = 1:50)
#
# pbmc.rna <- readRDS("pbmc_10k_v3.rds")
# pbmc.rna$tech <- "rna"
#
# transfer.anchors <- FindTransferAnchors(reference = pbmc.rna, query = pbmc.atac, features = VariableFeatures(object = pbmc.rna),
#     reference.assay = "RNA", query.assay = "ACTIVITY", reduction = "cca")
#
# celltype.predictions <- TransferData(anchorset = transfer.anchors, refdata = pbmc.rna$celltype,
#     weight.reduction = pbmc.atac[["lsi"]])
# pbmc.atac <- AddMetaData(pbmc.atac, metadata = celltype.predictions)
#
# pbmc.atac.filtered <- subset(pbmc.atac, subset = prediction.score.max > 0.5)
# pbmc.atac.filtered$predicted.id <- factor(pbmc.atac.filtered$predicted.id, levels = levels(pbmc.rna))  # to make the colors match
#
# DefaultAssay(pbmc.atac.filtered)  = 'ACTIVITY'
#
# atac_data = GetAssayData(pbmc.atac.filtered)
# atac_label = as.matrix(pbmc.atac.filtered$predicted.id)
#
#
# # p1 <- DimPlot(pbmc.atac, reduction = "umap") + NoLegend() + ggtitle("scATAC-seq")
#
# write.table(atac_data, file='atac_data.csv', sep=',', row.names=TRUE, col.names=TRUE,quote=FALSE)
# write.table(atac_label, file='atac_label.csv', sep=',', row.names=TRUE, col.names=TRUE,quote=FALSE)