library(scmap)
library(scater)
library(Seurat)

main <- function(path, ref_key, query_key){
    print(path)
    ref_data = paste('data_', ref_key, '.csv', sep='')
    ref_label = paste('label_', ref_key, '.csv', sep='')
    query_data = paste('data_', query_key, '.csv', sep='')
    query_label = paste('label_', query_key, '.csv', sep='')

    data1 = t(as.data.frame(read.csv(paste(path, 'ref', ref_data, sep='/'), row.names=1))) # 变为gene * cell
    data2 = t(as.data.frame(read.csv(paste(path, 'query', query_data, sep='/'), row.names=1)))
    print(dim(data1))
    print(dim(data2))

    label1 = as.data.frame(read.csv(paste(path, 'ref', ref_label, sep='/')))
    rownames(label1) = colnames(data1)
    label2 = as.data.frame(read.csv(paste(path, 'query', query_label, sep='/')))
    rownames(label2) = colnames(data2)

    rownames(data1) = rownames(data2)
    colnames(label1) = c('cell_type1')
    colnames(label2) = c('type')

    print("Finish reading data")


    # reference data
    # ref_obj@assays$RNA@data 代表做完Norm之后的data
    ref_sce <- SingleCellExperiment(assays = list(normcounts = as.matrix(data1)), colData = label1)
    logcounts(ref_sce) <- log2(normcounts(ref_sce) + 1)
    rowData(ref_sce)$feature_symbol <- rownames(ref_sce)
    ref_sce <- ref_sce[!duplicated(rownames(ref_sce)), ]

    query_sce <- SingleCellExperiment(assays=list(normcounts = as.matrix(data2)))
    logcounts(query_sce) <- log2(normcounts(query_sce) + 1)
    rowData(query_sce)$feature_symbol <- rownames(query_sce)


    # feature selection
    ref_sce <- selectFeatures(ref_sce, suppress_plot = TRUE)


    # scmap Cluster
    ref_sce <- indexCluster(ref_sce)

    # Threshold 默认是0.7
    scmapCluster_results <- scmapCluster(projection = query_sce,
                                        index_list = list(yan = metadata(ref_sce)$scmap_cluster_index),
                                        threshold=0)
    pred <- scmapCluster_results$scmap_cluster_labs
    write.table(pred, file="scmap_preds.csv", sep=',', row.names=TRUE, col.names=TRUE,quote=FALSE)

#
#     print(length(pred))
#     print(length(query_obj$type))
    match <- (pred==label2$type)
    print(pred)
    acc = sum(match) / length(match)

    return (acc)
}
final_acc <- c()
path = '../experiment/species/gse_mouse_human/data'
acc = main(path, '1', '1')
# final_acc<-append(final_acc, acc)
# path = '../experiment/platform/85241_84133/data'
# acc = main(path, '1', '1')
final_acc<-append(final_acc, acc)

print(final_acc)