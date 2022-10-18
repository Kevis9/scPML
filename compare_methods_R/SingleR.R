library(scmap)
library(scater)
library(Seurat)
library(SingleR)

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
    print("xxx")
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
        return(data)
        }
    )
    ref_obj <- objs1[[1]]
    query_obj <- objs1[[2]]


    # singler
    ref_se <- SummarizedExperiment(assays=list(counts = ref_obj@assays$RNA@counts, logcounts = ref_obj@assays$RNA@data))



    ref_se@colData$label.main <- as.matrix(label1)
    query_sce <- SingleCellExperiment(assays=list(counts=query_obj@assays$RNA@counts, logcounts=query_obj@assays$RNA@data))
    # query_sce <- logNormCounts(query_sce)
    print(dim(ref_obj@assays$RNA@counts))

    # quantile 0.8是默认
    query_pred <- SingleR(test = query_sce, ref = ref_se, labels = ref_se@colData$label.main,  quantile = 0.8)
    print("Single R finish")
    pred <- query_pred$labels
#     print(length(pred))
#     print(query_obj$type)
    match <- (pred==query_obj$type)
    write.table(pred, file="singler_preds.csv", sep=',', row.names=TRUE, col.names=TRUE,quote=FALSE)

    acc = sum(match)/length(match)
    return (acc)
}
#
# final_acc <- c()
# path = '../experiment/platform/task_seq_all/data'
# acc = main(path, '1', '5')
# # final_acc<-append(final_acc, acc)
# # path = '../experiment/platform/85241_84133/data'
# # acc = main(path, '1', '1')
# final_acc<-append(final_acc, acc)
#
# print(final_acc)

# library(scmap)
# library(scater)
# library(Seurat)
# library(SingleR)
#
# main <- function(path, ref_key, query_key){
#
#     print(path)
#     ref_data = paste('data_', ref_key, '.csv', sep='')
#     ref_label = paste('label_', ref_key, '.csv', sep='')
#     query_data = paste('data_', query_key, '.csv', sep='')
#     query_label = paste('label_', query_key, '.csv', sep='')
#
#     data1 = t(as.data.frame(read.csv(paste(path, 'ref', ref_data, sep='/'), row.names=1))) # 变为gene * cell
#     data2 = t(as.data.frame(read.csv(paste(path, 'query', query_data, sep='/'), row.names=1)))
#     print(dim(data1))
#     print(dim(data2))
#
#     label1 = as.data.frame(read.csv(paste(path, 'ref', ref_label, sep='/')))
#
#     rownames(label1) = colnames(data1)
#
#     label2 = as.data.frame(read.csv(paste(path, 'query', query_label, sep='/')))
#
#     rownames(label2) = colnames(data2)
#
#     rownames(data1) = rownames(data2)
#
#     colnames(label1) = c('cell_type1')
#     colnames(label2) = c('type')
#
#
#     print("Finish reading data")
#
#
#     # quantile 0.8是默认
#     query_pred <- SingleR(test = as.matrix(data2), ref = as.matrix(data1), labels = as.matrix(label1),  quantile = 0.8)
#     print("Single R finish")
#     pred <- query_pred$labels
#
#     match <- (pred==label2$type)
#     write.table(pred, file="singler_preds.csv", sep=',', row.names=TRUE, col.names=TRUE,quote=FALSE)
#
#     acc = sum(match)/length(match)
#     return (acc)
# }

final_acc <- c()
path = '../experiment/species/gse_mouse_human/data'
acc = main(path, '1', '1')
# final_acc<-append(final_acc, acc)
# path = '../experiment/platform/85241_84133/data'
# acc = main(path, '1', '1')
final_acc<-append(final_acc, acc)

print(final_acc)