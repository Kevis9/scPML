library(scmap)
library(scater)


path = '../experiment/platform/task_seq_all/data'
print(path)
data1 = t(as.data.frame(read.csv(paste(path, 'ref', 'data_1.csv', sep='/'), row.names=1))) # 变为gene * cell
data2 = t(as.data.frame(read.csv(paste(path, 'query', 'data_3.csv', sep='/'), row.names=1)))
print(dim(data1))
print(dim(data2))

label1 = as.data.frame(read.csv(paste(path, 'ref', 'label_1.csv', sep='/')))
rownames(label1) = colnames(data1)
label2 = as.data.frame(read.csv(paste(path, 'query', 'label_3.csv', sep='/')))
rownames(label2) = colnames(data2)

rownames(data1) = rownames(data2)
colnames(label1) = c('cell_type1')
colnames(label2) = c('type')

print("Finish reading data")
# reference data
ref_sce <- SingleCellExperiment(assays = list(normcounts = as.matrix(data1)), colData = label1)
# 标准化

# logcounts(ref_sce) <- log2(normcounts(ref_sce) + 1)
rowData(ref_sce)$feature_symbol <- rownames(ref_sce)
ref_sce <- ref_sce[!duplicated(rownames(ref_sce)), ]


# Feature selection
ref_sce <- selectFeatures(ref_sce, suppress_plot = FALSE)

# scmap cluster
# ref_sce <- indexCluster(ref_sce)

# scamp cell
ref_sce <- indexCell(ref_sce)

# Query Dataset
query_sce <- SingleCellExperiment(assays = list(normcounts = as.matrix(data2)))
# logcounts(query_sce) <- log2(normcounts(query_sce) + 1)
rowData(query_sce)$feature_symbol <- rownames(query_sce)
query_sce <- query_sce[!duplicated(rownames(query_sce)), ]
# query_sce <- selectFeatures(query_sce, suppress_plot = FALSE)

# Cluster Projection
# scmapCluster_results <- scmapCluster(
#   projection = ref_sce,
#   index_list = list(
#     ref = metadata(ref_sce)$scmap_cluster_index
#   )
# )

# Cell projection
nearest_neighbours <- scmap::scmapCell(projection=query_sce,
                                       index_list = list(ref = metadata(ref_sce)$scmap_cell_index))

scmap_cell_metadata <- colData(ref_sce)
colnames(scmap_cell_metadata) <- "celltypes"
mode_label <- function(neighbours, metadata=scmap_cell_metadata$celltypes) {
  freq <- table(metadata[neighbours])
  label <- names(freq)[which(freq == max(freq))]
  if (length(label) > 1) {return("Unassigned")}
  return(label)
}
# Apply these labels to the query cells
scmap_cell_labs <- apply(nearest_neighbours$ref$cells, 2, mode_label)
# Add the labels to the query object
# colData(query_sce)$scmap_cell <- scmap_cell_labs
pred <- scmap_cell_labs

# pred = scmapCluster_results$combined_labs

print(length(pred))
match <- (pred==label1)
print(sum(match)/length(match))
