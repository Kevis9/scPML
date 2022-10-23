library(Seurat)
library(CHETAH)
library(SingleR)
library(scmap)
library(conos)
library(pagoda2)

read_data <- function(path) {
    # return matrix
    data = as.matrix(read.csv(path, row.names=1))
    return (data)
}

read_label <- function(path) {
    #return matrix
    label = as.matrix(read.csv(path))
    return (label)
}

read_ref_query_data_label <- function(path, ref_key, query_key) {

    ref_data_path = paste(path, 'ref', paste('data_', ref_key, '.csv', sep=''), sep='/')
    ref_label_path = paste(path, 'ref', paste('label_', ref_key, '.csv', sep=''), sep='/')
    query_data_path = paste(path, 'query', paste('data_', query_key, '.csv', sep=''), sep='/')
    query_label_path = paste(path, 'query', paste('label_', query_key, '.csv', sep=''), sep='/')

    ref_data = t(read_data(ref_data_path)) # gene x cell
    ref_label = read_label(ref_label_path)
    query_data = t(read_data(query_data_path)) # gene x cell
    query_label = read_label(query_label_path)

    rownames(ref_data) = rownames(query_data)
    rownames(ref_label) = colnames(ref_data)
    rownames(query_label) = colnames(query_data)
    colnames(ref_label) = c('type')
    colnames(query_label) = c('type')

    return (list(ref_data, query_data, ref_label, query_label))

}

acc_score <- function(pred, true_label){
    prediction.match <- pred == true_label
    acc = sum(prediction.match)/length(prediction.match)
}

save_prediction <- function(pred, path) {
    write.table(predictions$predicted.id, file=path, sep=',', row.names=TRUE, col.names=TRUE,quote=FALSE)
}

seurat_pca_pred <- function(ref_data, query_data, ref_label, query_label) {
    # all input data must be matrix
    # reurn prediction (matrix)
    ref_data = as.data.frame(ref_data)
    query_data = as.data.frame(query_data)
    ref_label = as.data.frame(ref_label)
    query_label = as.data.frame(query_label)



    object1 <- CreateSeuratObject(
            counts=ref_data,
            project = "reference",
            assay = "RNA",
#             min.cells = 0,
#             min.features = 0,
#             names.field = 1,
#             names.delim = "_",
            meta.data = ref_label)

    object2 <- CreateSeuratObject(
        counts=query_data,
        project = "query",
        assay = "RNA",
#         min.cells = 0,
#         min.features =0,
#         names.field = 1,
#         names.delim = "_",
        meta.data = query_label
        )


    objs <- list(object1,object2)
    objs1 <- lapply(objs,function(data){
        data <- NormalizeData(data)
        data <- FindVariableFeatures(data,
                                       selection.method = "vst",
                                       nfeatures=2000)
        return(data)})

    reference.object <- objs1[[1]];
    query.object <- objs1[[2]]
#     reference.object <- ScaleData(reference.object, verbose = FALSE)
#     reference.object <- RunPCA(reference.object, npcs = 30, verbose = FALSE)

    reference.anchors <- FindTransferAnchors(reference = reference.object, query = query.object, dims = 1:30)
    predictions <- TransferData(anchorset = reference.anchors, refdata = as.factor(reference.object$type), dims = 1:30)

    return (as.matrix(predictions$predicted.id))
}

single_r_pred <- function(ref_data, query_data, ref_label) {
    # all input must be matrix
    # Single R can take matrix as input, reference data must be log-normalized
#     ref_sce = SingleCellExperiment(assays = list(normcounts = as.matrix(ref_data)))

#     ref_data <- as.matrix(log2(SingleCellExperiment::normcounts(ref_sce) + 1))

    # ref_data <- scater::logNormCounts(ref_data)
    pred <- SingleR(test = query_data, ref = ref_data, labels = ref_label)
    pred <- pred$labels
    return (pred)
}

scmap_pred <- function(ref_data, query_data, ref_label) {
    # all input must be matrix
    colnames(ref_label) = c('cell_type1')
    ref_sce <- SingleCellExperiment(assays = list(normcounts = as.matrix(ref_data)), colData = as.data.frame(ref_label))
    logcounts(ref_sce) <- log2(SingleCellExperiment::normcounts(ref_sce) + 1)
    rowData(ref_sce)$feature_symbol <- rownames(ref_sce)
    ref_sce <- ref_sce[!duplicated(rownames(ref_sce)), ]

    query_sce <- SingleCellExperiment(assays=list(normcounts = as.matrix(query_data)))
    logcounts(query_sce) <- log2(normcounts(query_sce) + 1)
    rowData(query_sce)$feature_symbol <- rownames(query_sce)


    # feature selection, default will be 500 genes being selected
    ref_sce <- selectFeatures(ref_sce, suppress_plot = TRUE)


    # scmap Cluster
    ref_sce <- scmap::indexCluster(ref_sce)


    scmapCluster_results <- scmapCluster(projection = query_sce,
                                         index_list = list(yan = metadata(ref_sce)$scmap_cluster_index),
                                         threshold=0)
    pred <- scmapCluster_results$scmap_cluster_labs
    return (as.matrix(pred))

}

chetah_pred <- function(ref_data, query_data, ref_label) {
    # all input must be matrix
    ref_ct = as.data.frame(ref_label)
    colnames(ref_ct) = c('celltypes')

    reference <- SingleCellExperiment(assays = list(counts = ref_data),
                                         colData = ref_ct)
    assay(reference, "counts") <- apply(assay(reference, "counts"), 2, function(column) log2((column/sum(column) * 100000) + 1))
    # aka query
    input <- SingleCellExperiment(assays = list(counts = query_data))


    input <- CHETAHclassifier(input = input, ref_cells = reference)
    ## Extract celltypes:
    pred <- input$celltype_CHETAH
    return (as.matrix(pred))
}

main <- function(path, ref_key, query_key, method){

    # return accuracy
    print(path)
    data = read_ref_query_data_label(path, ref_key, query_key)

    ref_data = data[[1]]
    query_data = data[[2]]
    ref_label = data[[3]]
    query_label = data[[4]]

    print("数据读取完成")
    if(method == 'seurat') {
        pred = seurat_pca_pred(ref_data, query_data, ref_label, query_label)

        acc = acc_score(pred, query_label)
    }
    if(method == 'singler') {
        pred = single_r_pred(ref_data, query_data, ref_label)
        acc = acc_score(pred, query_label)
    }
    if(method == 'scmap') {
        pred = scmap_pred(ref_data, query_data, ref_label)
        acc = acc_score(pred, query_label)
    }
    if(method == 'chetah') {
        pred = chetah_pred(ref_data, query_data, ref_label)
        acc = acc_score(pred, query_label)
    }

    return (acc)
}
final_acc = c()

path = '../experiment/species/gse_emtab_common_type/human_mouse/data'
acc = c(
        main(path, '1', '1', 'seurat'),
        main(path, '1', '1', 'singler'),
        main(path, '1', '1', 'scmap'),
        main(path, '1', '1', 'chetah')
)
acc = setNames(acc, c('seurat', 'singler', 'scamp', 'chetah'))

print(acc)
