library(Seurat)
# ' @param count.list
# ' @param label.list


# count.list <- readRDS('./example_data/count.list.RDS')
# label.list <- readRDS('./example_data/label.list.RDS')
# print(dim(count.list[[1]]))
# print(dim(count.list[[2]]))


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
    colnames(label1) = c('type')
    colnames(label2) = c('type')

    object1 <- CreateSeuratObject(
            counts=data1,
            project = "reference",
            assay = "RNA",
            min.cells = 0,
            min.features = 0,
            names.field = 1,
            names.delim = "_",
            meta.data = label1)

    object2 <- CreateSeuratObject(
        counts=data2,
        project = "query",
        assay = "RNA",
        min.cells = 0,
        min.features =0,
        names.field = 1,
        names.delim = "_",
        meta.data = label2)


    objs <- list(object1,object2)
    objs1 <- lapply(objs,function(indrop){
        indrop <- NormalizeData(indrop)
        indrop <- FindVariableFeatures(indrop,
                                       selection.method = "vst",
                                       nfeatures=2000)
        return(indrop)})

    reference.object <- objs1[[1]]; query.object <- objs1[[2]]
    reference.object <- ScaleData(reference.object, verbose = FALSE)
    reference.object <- RunPCA(reference.object, npcs = 30, verbose = FALSE)

    reference.anchors <- FindTransferAnchors(reference = reference.object, query = query.object, dims = 1:30)
    predictions <- TransferData(anchorset = reference.anchors, refdata = as.factor(reference.object$type), dims = 1:30)

    prediction.match <- predictions$predicted.id == query.object$type

    acc = sum(prediction.match)/length(prediction.match)
    return (acc)
}

final_acc <- c()
path = '../experiment/platform/emtab5016_gse84133/data'
acc = main(path, '1', '1')
print(acc)