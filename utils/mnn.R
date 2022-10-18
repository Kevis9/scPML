library(Seurat)
library(SeuratWrappers)
library(ggplot2)
library(batchelor)
# library(patchwork)


main <- function(path, ref_key, query_key){
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
    tech_arr = c()
    for(i in 1:dim(label1)[1]){
        tech_arr = append(tech_arr, 'reference')
    }
    label1$tech = tech_arr

    tech_arr = c()
    for(i in 1:dim(label2)[1]){
        tech_arr = append(tech_arr, 'query')
    }
    label2$tech = tech_arr

    out <- mnnCorrect(as.matrix(data1), as.matrix(data2))
    return (out)
    object1 <- CreateSeuratObject(
            counts=data1,
            meta.data = label1)

    object2 <- CreateSeuratObject(
        counts=data2,
        meta.data = label2)


    objs <- list(object1,object2)
    objs1 <- lapply(objs,function(indrop){
        indrop <- NormalizeData(indrop)
        indrop <- FindVariableFeatures(indrop,
                                       selection.method = "vst",
                                       nfeatures=2000)
        return(indrop)})
    scRNA <- RunFastMNN(object.list = objs)
    scRNA <- RunUMAP(scRNA, reduction = "mnn", dims = 1:30)

    data = cbind(data1, data2)
    label = rbind(label1, label2)
    origin <- CreateSeuratObject(
            counts=data,
            meta.data = label)
    # origin <- merge(objs[[1]], objs[[2]])
#     origin <- NormalizeData(origin, verbose=FALSE)
#     origin <- FindVariableFeatures(origin, selection.method = "vst", nfeatures=2000)
#     origin <- ScaleData(origin, verbose=FALSE)
#     origin <- RunPCA(origin, npcs=30, verbose = FALSE)
#     origin <- RunUMAP(origin, dims = 1:30)
#     origin <- RunUMAP(scRNA, reduction = "mnn", dims = 1:30)
#     scRNA <- FindNeighbors(scRNA, reduction = "mnn", dims = 1:30)
#     scRNA <- FindClusters(scRNA)
#     objs1 <- RunUMAP(objs1, reduction = "mnn", dims = 1:30)
    print(scRNA)
    p1 <- DimPlot(scRNA, group.by = "tech", pt.size=0.1) +
        ggtitle("Integrated by fastMNN")
    p2 <- DimPlot(origin, group.by="tech", pt.size=0.1) +
        ggtitle("No integrated")
    p = p1 + p2
#     p = p1
    ggsave('fastMNN.png', p, width=8, height=4)
    return (scRNA)
}

path = '../experiment/species/gse_common_type/mouse_human/data'
scRNA = main(path, '1', '1')
