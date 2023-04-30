
suppressMessages(library(Seurat));
suppressMessages(library(entropy))

GenerateGraph <- function(data,k_neighbor){
    obj <- CreateSeuratObject(counts=data,project = "1",assay = "data",
                                  min.cells = 0,min.features = 0,
                                  names.field = 1,names.delim = "_")


    obj <- NormalizeData(obj,verbose=F)
    # 取2000没关系，最后是保留最小的那个
    obj <- FindVariableFeatures(obj,
                            selection.method = "vst",
                            nfeatures = 2000,verbose=F)
    # obj <- ScaleData(obj,features=rownames(obj),verbose=FALSE)
    # 这里能看到PCA是直接作用在raw data的feature上的，而不是HVGs上, scGCN真的有点毛病
    # obj <- RunPCA(obj, features=rownames(obj), verbose = FALSE)
    # Intra-data graph
    d2.list <- list(obj, obj)
    d2.nn <- FindIntegrationAnchors(object.list =d2.list,k.anchor=k_neighbor,verbose=F)
    d2.arc=d2.nn@anchors

    d2.arc1=cbind(d2.arc[d2.arc[,4]==1,1],d2.arc[d2.arc[,4]==1,2],d2.arc[d2.arc[,4]==1,3])
    d2.grp=d2.arc1[d2.arc1[,3]>0,1:2]-1
    return (d2.grp)
}

read_data <- function(path) {
    # return matrix
    data = as.matrix(read.csv(path, row.names=1))
    return (data)
}

data = read_data('mat_path_1_1.csv')
graph = GenerateGraph(data, k_neighbor=5)
write.csv(graph,file='graph.csv',quote=F,row.names=T)