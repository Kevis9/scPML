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
#         data <- FindVariableFeatures(data,
#                                        selection.method = "vst",
#                                        nfeatures=500)
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

    # scamp cell
#     ref_sce <- indexCell(ref_sce)



    query_sce <- SingleCellExperiment(assays=list(normcounts = as.matrix(query_obj@assays$RNA@counts)))
    logcounts(query_sce) <- query_obj@assays$RNA@data
    rowData(query_sce)$feature_symbol <- rownames(query_sce)
#     query_sce <- selectFeatures(query_sce, suppress_plot = FALSE)
       # # Cell projection
#     nearest_neighbours <- scmap::scmapCell(projection=query_sce,
#                                        index_list = list(ref = metadata(ref_sce)$scmap_cell_index))
#
#     scmap_cell_metadata <- colData(ref_sce)
#     colnames(scmap_cell_metadata) <- "celltypes"
#     mode_label <- function(neighbours, metadata=scmap_cell_metadata$celltypes) {
#       freq <- table(metadata[neighbours])
#       label <- names(freq)[which(freq == max(freq))]
#       if (length(label) > 1) {return("Unassigned")}
#       return(label)
#     }
#     # Apply these labels to the query cells
#     scmap_cell_labs <- apply(nearest_neighbours$ref$cells, 2, mode_label)
#     # Add the labels to the query object
#     # colData(query_sce)$scmap_cell <- scmap_cell_labs
#     pred <- scmap_cell_labs

    scmapCluster_results <- scmapCluster(projection = query_sce, index_list = list(ref = metadata(ref_sce)$scmap_cluster_index),threshold=0.6)
    pred <- scmapCluster_results$scmap_cluster_labs
#
#     print(length(pred))
#     print(length(query_obj$type))
    match <- (pred==query_obj$type)
    print(pred)
    acc = sum(match) / length(match)

    return (acc)
}

final_acc <- c()
path = '../experiment/platform/task_dropseq_all/data'
acc = main(path, '1', '1')
print(acc)
final_acc<-append(final_acc, acc)
# path = '../experiment/species/gsehuman_gsemouse/data'
acc = main(path, '1', '2')
print(acc)
final_acc<-append(final_acc, acc)
# path = '../experiment/species/gsemouse_emtab/data'
acc = main(path, '1', '3')
print(acc)
final_acc<-append(final_acc, acc)
# path = '../experiment/species/emtab_gsemouse/data'
acc = main(path, '1', '4')
final_acc<-append(final_acc, acc)
acc = main(path, '1', '5')
final_acc<-append(final_acc, acc)
print(final_acc)