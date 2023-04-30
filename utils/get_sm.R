suppressMessages(library(parallelDist))
suppressMessages(library(SNFtool)) # SNF;spectralClustering
suppressMessages(library(GSEABase)) # getGmt, load pathway information
suppressMessages(library(AUCell)) # AUCell, pathway scoring method
suppressMessages(library(Seurat))

# 为GSE84133_RAW设置的load_matrix
load_matrix_for_GSE <- function(path){
  # check.names 去掉X
  expr_matrix = read.csv(path,check.names=FALSE, row.names=1)
  expr_matrix = as.matrix(expr_matrix)
  return(expr_matrix)
}

# load pathway
load_pathway <- function(path,name){
  gSet=getGmt(paste(path,name,sep='/'))
  return(gSet)
}


# pathway scoring: AUCell
pathway_scoring <- function(gSet, mat_gene){
  cells_rankings <- AUCell_buildRankings(mat_gene, plotStats=TRUE)
  cells_AUC <- AUCell_calcAUC(gSet, cells_rankings, aucMaxRank = ceiling(0.05 * nrow(cells_rankings)))
  aucMatrix <- getAUC(cells_AUC)
  aucMatrix = aucMatrix[rowSums(aucMatrix)>0.0,]
  return(aucMatrix)
}

#
clean_sets <- function(gSet){
  min.size = 10; max.size = 500
  len_s = sapply(gSet, function(x) length(geneIds(x)))
  idx = (len_s > min.size)&(len_s<max.size)
  gSet = gSet[idx]
  return(gSet)
}

# adding pathway: SNF
integrating_pathway <- function(mat_gene, mat_path){
  K = 10; # number of neighbors, usually (10~30)
  alpha = 0.5; # hyperparameter, usually (0.3~0.8)
  T = 20; # Number of Iterations, usually (10~20)

  # mat_gene 预处理
#   mat_gene = standardNormalization(mat_gene)
#   mat_gene = as.matrix(parDist(as.matrix(mat_gene), method='euclidean', threads=400))
#   # mat_gene = (dist2(as.matrix(mat_gene),as.matrix(mat_gene)))^(1/2)
#   mat_gene = affinityMatrix(mat_gene, K, alpha)

  # 尝试对mat path 做一个findvariavle gene的操作, 可关闭
#   obj <- CreateSeuratObject(counts=mat_path)
#   obj <- FindVariableFeatures(obj,
#                                 selection.method = "vst",
#                                 nfeatures = 30, verbose=F)
#   mat_path = obj[obj@assays$RNA@var.features,]@assays$RNA@data

  mat_path = t(mat_path)
  mat_path = standardNormalization(mat_path)
  mat_path = as.matrix(parDist(as.matrix(mat_path), method='euclidean'), threads=400) # 并行的方式去加速计算dist 代表细胞间的correlation
  mat_path = affinityMatrix(mat_path, K, alpha)

  W = mat_path
  # 这里把mat_gene和mat_path做一个融合然后返回
#   W = SNF(list(mat_path, mat_gene), K, T)

  return(W)

  # return(mat_path)
}

mat_gene_reduction <- function(mat_gene) {
    # 传入 gene * cell
    # 对mat_gene做一个标准Seurat处理
    obj <- CreateSeuratObject(counts=mat_gene)
    obj <- FindVariableFeatures(obj,
                                selection.method = "vst",
                                nfeatures = 2000,verbose=F)
    obj <- ScaleData(obj,verbose=FALSE)
    obj <- RunPCA(obj, verbose = FALSE)
    # 得到经过 HVGs -> PCA之后的cell * gene (mat_gene)
    mat_gene.reduction = obj@reductions$pca@cell.embeddings
    # 返回 cell * gene
    return (mat_gene.reduction)
}

get_mat_path <- function(mat_gene, paPath, human_pathway_name) {
    gSet = load_pathway(paPath, paste(human_pathway_name,'.gmt',sep=''))
    gSet = subsetGeneSets(gSet, rownames(mat_gene)) #AUCell
    gSet = clean_sets(gSet) # min.size = 5; max.size = 500
    # pathway scoring: AUCell
    mat_path = pathway_scoring(gSet, mat_gene)
    return (mat_path)
}

get_cell_similarity_matrix<-function(mat_path){
  K = 10; # number of neighbors, usually (10~30)
  alpha = 0.5; # hyperparameter, usually (0.3~0.8)
  T = 20; # Number of Iterations, usually (10~20)

  # 输入是pathway * cell
  mat_path = t(mat_path)
  mat_path = standardNormalization(mat_path)
  mat_path = as.matrix(parDist(as.matrix(mat_path), method='euclidean'), threads=400) # 并行的方式去加速计算dist 代表细胞间的correlation
  mat_path = affinityMatrix(mat_path, K, alpha)

  W = mat_path
  # 这里把mat_gene和mat_path做一个融合然后返回
#   W = SNF(list(mat_path, mat_gene), K, T)

  return (W)

}

get_cell_similarity_matrix_by_genes <- function(mat_gene) {
    mat_gene.reduction = mat_gene_reduction(mat_gene)
    print("dim of mat_gene.reduction")
    print(dim(mat_gene.reduction))
    K = 10; # number of neighbors, usually (10~30)
    alpha = 0.5; # hyperparameter, usually (0.3~0.8)
    T = 20; # Number of Iterations, usually (10~20)

    # mat_gene 预处理

    mat_gene.reduction = standardNormalization(mat_gene.reduction)
    mat_gene.reduction = as.matrix(parDist(as.matrix(mat_gene.reduction), method='euclidean', threads=50))
    # mat_gene = (dist2(as.matrix(mat_gene),as.matrix(mat_gene)))^(1/2)
    W = affinityMatrix(mat_gene.reduction, K, alpha)
    print("dim of W")
    print(dim(W))
    return (W)
}


main <- function(data_path, data_num=1, view_num=4, pathway_name) {
    for(i in 1:data_num) {
      mat_name = paste('data_', i, '.csv', sep='')
      mat_gene = load_matrix_for_GSE(paste(data_path, mat_name, sep='\\'))
      mat_gene = t(mat_gene) # 对于(cell*genes)格式的数据，先做一次转置变成 gene * cell

      for(j in 1:view_num) {
          save_path_W = paste(data_path, paste(paste('sm_', i, '_', sep=''), j, '.csv',sep=''), sep='\\')
          save_path_mat_path = paste(data_path, paste(paste('mat_path_', i, '_', sep=''), j, '.csv',sep=''), sep='\\')

          mat_path = get_mat_path(mat_gene, paPath, pathway_name[j])
          W = get_cell_similarity_matrix(mat_path)

          print("Save the W (integrated) matrix")
          write.csv(W, save_path_W)
#           print("Save mat_path")
#           write.csv(mat_path, save_path_mat_path)
      }

      # 按照gene feature来获取特征
      mat_sm_gene = get_cell_similarity_matrix_by_genes(mat_gene)
      write.csv(mat_sm_gene, paste(data_path, paste('sm_', i, '_', view_num+1, '_.csv', sep=''), sep='/'))
    }
}


paPath = "E:\\yuanhuang\\kevislin\\data\\pathway\\new_human"
mouse_pathway_name = c('KEGG', 'Reactome', 'Wikipathways', 'biase')
human_pathway_name = c('KEGG', 'Reactome', 'Wikipathways', 'yan', 'inoh', 'pathbank')
# human_pathway_name = mouse_pathway_name
args = commandArgs(trailingOnly = TRUE)
base_path = args[[1]]

pathway_name = human_pathway_name
# ref
data_path = paste(base_path, 'raw_data', 'ref', sep='\\')
main(data_path, data_num=1, view_num=4, pathway_name)

# query
data_path = paste(base_path, 'raw_data', 'query', sep='\\')
main(data_path, data_num=1, view_num=4, pathway_name)