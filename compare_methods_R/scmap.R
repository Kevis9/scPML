library(scmap)
library(scater)
library(Seurat)

path = '../experiment/platform/task_seq_all/data'
print(path)
data1 = t(as.data.frame(read.csv(paste(path, 'ref', 'data_1.csv', sep='/'), row.names=1))) # 变为gene * cell
data2 = t(as.data.frame(read.csv(paste(path, 'query', 'data_1.csv', sep='/'), row.names=1)))
print(dim(data1))
print(dim(data2))

label1 = as.data.frame(read.csv(paste(path, 'ref', 'label_1.csv', sep='/')))
rownames(label1) = colnames(data1)
label2 = as.data.frame(read.csv(paste(path, 'query', 'label_1.csv', sep='/')))
rownames(label2) = colnames(data2)

rownames(data1) = rownames(data2)
colnames(label1) = c('cell_type1')
colnames(label2) = c('type')

# data1 = as.matrix(data1)
# data2 = as.matrix(data2)
# label1 = as.matrix(label1)
# label2 = as.matrix(label2)

print("Finish reading data")

# 先封装成Seurat object
ref_obj <- CreateSeuratObject(
        counts=data1,
        project = "reference",
        assay = "RNA",
        min.cells = 0,
        min.features = 0,
        names.field = 1,
        names.delim = "_",
        meta.data = label1)

query_obj <- CreateSeuratObject(
    counts=data2,
    project = "query",
    assay = "RNA",
    min.cells = 0,
    min.features =0,
    names.field = 1,
    names.delim = "_",
    meta.data = label2)


objs <- list(ref_obj, query_obj)
objs1 <- lapply(objs,function(data){
    data <- NormalizeData(data)
    return(data)
    }
)
ref_obj <- objs1[[1]]
query_obj <- objs1[[2]]


# reference data
# ref_obj@assays$RNA@data 代表做完Norm之后的data
ref_sce <- SingleCellExperiment(assays = list(normcounts = as.matrix(ref_obj@assays$RNA@counts)), colData = label1)
logcounts(ref_sce) <-ref_obj@assays$RNA@data
rowData(ref_sce)$feature_symbol <- rownames(ref_sce)
ref_sce <- ref_sce[!duplicated(rownames(ref_sce)), ]

# feature selection
ref_sce <- selectFeatures(ref_sce, suppress_plot = FALSE)
# var_genes <- VariableFeatures(query_obj)
# print(var_genes)
# print(length(var_genes))
#
# # var_genes <- str_replace(var_genes, "-", "_")
# ref_sce <- setFeatures(ref_sce, var_genes)

# scmap Cluster
ref_sce <- indexCluster(ref_sce)

query_sce <- SingleCellExperiment(assays=list(normcounts = as.matrix(query_obj@assays$RNA@counts)))
logcounts(query_sce) <- query_obj@assays$RNA@data
rowData(query_sce)$feature_symbol <- rownames(query_sce)

scmapCluster_results <- scmapCluster(projection = query_sce, index_list = list(ref = metadata(ref_sce)$scmap_cluster_index))
pred <- scmapCluster_results$scmap_cluster_labs

print(length(pred))
print(length(query_obj$type))
match <- (pred==query_obj$type)
print('acc')
print(sum(match)/length(match))


# Feature selection
# ref_sce <- selectFeatures(ref_sce, suppress_plot = FALSE)

# scmap cluster
# ref_sce <- indexCluster(ref_sce)

# scamp cell
# ref_sce <- indexCell(ref_sce)

# Query Dataset
# query_sce <- SingleCellExperiment(assays = list(normcounts = as.matrix(data2)))
# # logcounts(query_sce) <- log2(normcounts(query_sce) + 1)
# rowData(query_sce)$feature_symbol <- rownames(query_sce)
# query_sce <- query_sce[!duplicated(rownames(query_sce)), ]
# # query_sce <- selectFeatures(query_sce, suppress_plot = FALSE)
#
# # Cluster Projection
# # scmapCluster_results <- scmapCluster(
# #   projection = ref_sce,
# #   index_list = list(
# #     ref = metadata(ref_sce)$scmap_cluster_index
# #   )
# # )
#
# # Cell projection
# nearest_neighbours <- scmap::scmapCell(projection=query_sce,
#                                        index_list = list(ref = metadata(ref_sce)$scmap_cell_index))
#
# scmap_cell_metadata <- colData(ref_sce)
# colnames(scmap_cell_metadata) <- "celltypes"
# mode_label <- function(neighbours, metadata=scmap_cell_metadata$celltypes) {
#   freq <- table(metadata[neighbours])
#   label <- names(freq)[which(freq == max(freq))]
#   if (length(label) > 1) {return("Unassigned")}
#   return(label)
# }
# # Apply these labels to the query cells
# scmap_cell_labs <- apply(nearest_neighbours$ref$cells, 2, mode_label)
# # Add the labels to the query object
# # colData(query_sce)$scmap_cell <- scmap_cell_labs
# pred <- scmap_cell_labs
#
# # pred = scmapCluster_results$combined_labs
#
# print(length(pred))
# match <- (pred==label1)
# print(sum(match)/length(match))
