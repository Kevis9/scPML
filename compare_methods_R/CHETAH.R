library(CHETAH)
library(Seurat)

main <- function(path, ref_key, query_key){
    print("start")
#     path = '../experiment/platform/task_seq_all/data'
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
    colnames(label1) = c('celltypes')
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

    # CHETAH does not require normalized input data, but the reference data has to be normalized beforehand.
    ref_counts = ref_obj@assays$RNA@data
    ref_ct = label1

    reference <- SingleCellExperiment(assays = list(counts = ref_counts),
                                         colData = ref_ct)
    input <- SingleCellExperiment(assays = list(counts = query_obj@assays$RNA@counts))


    input <- CHETAHclassifier(input = input, ref_cells = reference)
    ## Extract celltypes:
    pred <- input$celltype_CHETAH
    match <- (pred==label2)
    acc <- sum(match)/length(match)
    return (acc)
}

final_acc <- c()
path = '../experiment/platform/emtab5016_gse84133/data'
acc = main(path, '1', '1')
print(acc)
# final_acc<-append(final_acc, acc)
# # path = '../experiment/species/gsehuman_gsemouse/data'
# acc = main(path, '1', '2')
# print(acc)
# final_acc<-append(final_acc, acc)
# # path = '../experiment/species/gsemouse_emtab/data'
# acc = main(path, '1', '3')
# print(acc)
# final_acc<-append(final_acc, acc)
# # path = '../experiment/species/emtab_gsemouse/data'
# acc = main(path, '1', '4')
# final_acc<-append(final_acc, acc)
# acc = main(path, '1', '5')
# final_acc<-append(final_acc, acc)
# print(final_acc)