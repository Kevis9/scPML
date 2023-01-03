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
#     print(ref_data_path)
#     print(query_data_path)
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

seurat_pca_pred <- function(ref_data, query_data, ref_label, query_label, save_path=".") {
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
            meta.data = ref_label)

    object2 <- CreateSeuratObject(
        counts=query_data,
        project = "query",
        assay = "RNA",
        meta.data = query_label
        )


    objs <- list(object1,object2)
    objs1 <- lapply(objs,function(data){
        data <- NormalizeData(data)
        data <- FindVariableFeatures(data,
                                       selection.method = "vst",
                                       nfeature=2000)
        return(data)})

    reference.object <- objs1[[1]];
    query.object <- objs1[[2]]
    reference.object <- ScaleData(reference.object, verbose = FALSE)
    reference.object <- RunPCA(reference.object, npcs = 30, verbose = FALSE)
    reference.anchors <- FindTransferAnchors(reference = reference.object, query = query.object, dims = 1:30, reference.reduction = "pca")
    reference.object = RunUMAP(reference.object, dims = 1:30, reduction = "pca", return.model = TRUE)
#     predictions <- TransferData(anchorset = reference.anchors, refdata = as.factor(reference.object$type), dims = 1:30)
    query <-  MapQuery(anchorset = reference.anchors, reference = reference.object, query = query.object,
    refdata = list(celltype = "type"), reference.reduction = "pca", reduction.model = "umap")
    pred = query$predicted.celltype

    ref_umap_data = reference.object@reductions$umap@cell.embeddings
    query_umap_data = query@reductions$ref.umap@cell.embeddings

#     write.csv(ref_umap_data, paste(save_path, "ref_embeddings_2d.csv", sep='/'))
#     write.csv(query_umap_data, paste(save_path, "query_embeddings_2d.csv", sep='/'))
#     write.csv(ref_label,paste(save_path, "ref_label.csv", sep='/'), row.names=FALSE)
#     write.csv(pred, paste(save_path, "query_pred.csv", sep='/'), row.names=FALSE)
    return (as.matrix(pred))
}

seurat_cca_pred <- function(ref_data, query_data, ref_label, query_label, save_path=".") {
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
            meta.data = ref_label)

    object2 <- CreateSeuratObject(
        counts=query_data,
        project = "query",
        assay = "RNA",
        meta.data = query_label
        )


    objs <- list(object1,object2)
    objs1 <- lapply(objs,function(data){
        data <- NormalizeData(data)
        data <- FindVariableFeatures(data,
                                       selection.method = "vst",
                                       nfeature=2000)
        return(data)})

    reference.object <- objs1[[1]];
    query.object <- objs1[[2]]
    reference.object <- ScaleData(reference.object, verbose = FALSE)
    reference.object <- RunPCA(reference.object, npcs = 30, verbose = FALSE)
    reference.anchors <- FindTransferAnchors(reference = reference.object, query = query.object, dims = 1:30, reduction = 'cca')
    reference.object = RunUMAP(reference.object, dims = 1:30, reduction = "pca", return.model = TRUE)
#     predictions <- TransferData(anchorset = reference.anchors, refdata = as.factor(reference.object$type), dims = 1:30)
    query <-  MapQuery(anchorset = reference.anchors, reference = reference.object, query = query.object,
    refdata = list(celltype = "type"), reference.reduction = "cca", reduction.model = "umap")
    pred = query$predicted.celltype

    ref_umap_data = reference.object@reductions$umap@cell.embeddings
    query_umap_data = query@reductions$ref.umap@cell.embeddings

    write.csv(ref_umap_data, paste(save_path, "seurat_cca_ref_embeddings_2d.csv", sep='_'))
    write.csv(query_umap_data, paste(save_path, "seurat_cca_query_embeddings_2d.csv", sep='_'))
    write.csv(ref_label,paste(save_path, "seurat_cca_ref_label.csv", sep='_'), row.names=FALSE)
    write.csv(pred, paste(save_path, "seurat_cca_query_pred.csv", sep='_'), row.names=FALSE)
    return (as.matrix(pred))
}

conos_pred <- function(ref_data, query_data, ref_label) {
    cellannot = data.frame(
        cell_name = as.vector(colnames(ref_data)),
        cell_type = ref_label[, 1]
    )
#     rownames(cellannot) = c(1:dim(cellannot)[1])
    cellannot <- setNames(cellannot[,2], cellannot[,1])
    panel <- list(ref=ref_data, query=query_data)
    panel.preprocessed <- lapply(panel, basicSeuratProc)
    con <- Conos$new(panel.preprocessed, n.cores=1)
    con$buildGraph(k=30, k.self=5, space='PCA', ncomps=30, n.odgenes=2000, matching.method='mNN', metric='angular', score.component.variance=TRUE, verbose=TRUE)
    pred <- con$propagateLabels(labels = cellannot)$labels
    query_pred = pred[colnames(query_data)]
    return (as.matrix(query_pred))
}


single_r_pred <- function(ref_data, query_data, ref_label) {
    # all input must be matrix
    # Single R can take matrix as input, reference data must be log-normalized
    ref_sce = SingleCellExperiment(assays = list(normcounts = as.matrix(ref_data)))
    # Normalize reference data
#     ref_data <- as.matrix(log2(SingleCellExperiment::normcounts(ref_sce) + 1))

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
    # Normalize reference data
    assay(reference, "counts") <- apply(assay(reference, "counts"), 2, function(column) log2((column/sum(column) * 1e4) + 1))
    # aka query
    input <- SingleCellExperiment(assays = list(counts = query_data))


    input <- CHETAHclassifier(input = input, ref_cells = reference)
    ## Extract celltypes:
    pred <- input$celltype_CHETAH
    return (as.matrix(pred))
}

main <- function(path, ref_key, query_key, method, save_path){

    # return accuracy
    print(path)
    data = read_ref_query_data_label(path, ref_key, query_key)

    ref_data = data[[1]]
    query_data = data[[2]]
    ref_label = data[[3]]
    query_label = data[[4]]

    print("数据读取完成")
    if(method == 'seurat_pca') {
        pred = seurat_pca_pred(ref_data, query_data, ref_label, query_label, save_path)
        acc = acc_score(pred, query_label)
    }
    if(method == 'seurat_cca') {
        pred = seurat_cca_pred(ref_data, query_data, ref_label, query_label, save_path)
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
    if(method == 'conos') {
        pred = conos_pred(ref_data, query_data, ref_label)
        acc = acc_score(pred, query_label)
    }


    write.csv(pred, paste("result/", paste(paste(save_path, method, sep='_'), ".csv", sep=''), sep=''))
    return (acc)
}

final_acc = list()
platform_project = c('cel_seq_smart_seq',
            'cel_seq_10x_v3',
            'seq_well_smart_seq',
            'seq_well_drop_seq',
            'seq_well_10x_v3',
            'smart_seq_10x_v3',
            'indrop_drop_seq',
            'indrop_10x_v3',
            'indrop_smart_seq',
            'drop_seq_smart_seq',
            'drop_seq_10x_v3',
            '84133_5061'
            )

species_project = c(
    'gse/mouse_human',
    'gse/human_mouse',
    'gse_emtab/mouse_human',
    'mouse_combine'
#     'gsemouse_gse85241'
)
name=  c(
    'gsemouse_gsehuman',
    'gsehuman_gsemouse',
    'gsemouse_emtab',
    'gsemouse_combine'
)

project = species_project

path = '../experiment/species_v3'
save_path = '.' #这里暂时不需要

for(i in 1:length(project)){

    data_path = paste(path, project[i], 'raw_data', sep='/')
    acc = c(
#         main(data_path, '1', '1', 'seurat_pca', save_path),
        main(data_path, '1', '1', 'seurat_cca', name[i])
#         main(data_path, '1', '1', 'singler'),
#         main(data_path, '1', '1', 'scmap'),
#         main(data_path, '1', '1', 'chetah')
    )
    acc = as.matrix(acc)
    dim(acc) = rev(dim(acc))
    print(acc)
#     colnames(acc) = c('seurat_pca', 'seurat_cca','singler', 'scmap', 'chetah')
    final_acc[[length(final_acc)+1]] = acc
}
final_acc = do.call(rbind, final_acc)
rownames(final_acc) = project
# write.csv(final_acc, file='result/acc_species.csv')