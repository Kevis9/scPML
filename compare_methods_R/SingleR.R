library(scmap)
library(scater)
library(Seurat)
library(SingleR)

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

print(dim(ref_se))
print(dim(label1))
query_pred <- SingleR(test = query_sce, ref = ref_se, labels = ref_se@colData$label.main)
print("Single R finish")
pred <- query_pred$labels
print(length(pred))
print(query_obj$type)
match <- (pred==query_obj$type)
print(sum(match)/length(match))