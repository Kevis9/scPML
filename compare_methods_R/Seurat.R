library(Seurat)
# ' @param count.list
# ' @param label.list


# count.list <- readRDS('./example_data/count.list.RDS')
# label.list <- readRDS('./example_data/label.list.RDS')
# print(dim(count.list[[1]]))
# print(dim(count.list[[2]]))

path = 'experiment\\platform\\task_seq_10x'
print(path)
data1 = t(as.data.frame(read.csv(paste(path, 'ref_raw_data.csv', sep='\\'), row.names=1))) # 变为gene * cell
data2 = t(as.data.frame(read.csv(paste(path, 'query_raw_data.csv', sep='\\'), row.names=1)))
print(dim(data1))
print(dim(data2))
label1 = as.data.frame(read.csv(paste(path, 'ref_label.csv', sep='\\')))
rownames(label1) = colnames(data1)
label2 = as.data.frame(read.csv(paste(path, 'query_label.csv', sep='\\')))
rownames(label2) = colnames(data2)
rownames(data1) = rownames(data2)
colnames(label1) = c('type')
colnames(label2) = c('type')
# data1 = count.list[[1]]
# data2 = count.list[[2]]
# label1 = label.list[[1]]
# label2 = label.list[[2]]
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

print (sum(prediction.match)/length(prediction.match))


