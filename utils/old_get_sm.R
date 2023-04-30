# if (!require("BiocManager", quietly = TRUE))
#     install.packages("BiocManager", repos="https://mirrors.tuna.tsinghua.edu.cn/CRAN/")
# BiocManager::install("SNFtool")
# BiocManager::install("GSEABase")
# BiocManager::install("AUCell")
suppressMessages(library(parallelDist))
suppressMessages(library(SNFtool)) # SNF;spectralClustering
suppressMessages(library(GSEABase)) # getGmt, load pathway information
suppressMessages(library(AUCell)) # AUCell, pathway scoring method
suppressMessages(library(Seurat))

# 为GSE84133_RAW设置的load_matrix
load_matrix_for_GSE <- function(path){
#   expr_matrix = read.csv(path,check.names=FALSE) # 去掉X
#   # print(colnames(expr_matrix))
#   # print(rownames(expr_matrix))
#   expr_matrix = expr_matrix[,-1]#删掉第1列 cell id
#   expr_matrix = as.matrix(expr_matrix)
#     return(expr_matrix)
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
# 这是原来的做法
integrating_pathway <- function(mat_gene, mat_path){
  K = 10; # number of neighbors, usually (10~30)
  alpha = 0.5; # hyperparameter, usually (0.3~0.8)
  T = 20; # Number of Iterations, usually (10~20)

  # mat_gene 预处理
#   mat_gene = t(mat_gene)
#   mat_gene = standardNormalization(mat_gene)
#   mat_gene = as.matrix(parDist(as.matrix(mat_gene), method='euclidean', threads=400))
# #   mat_gene = (dist2(as.matrix(mat_gene),as.matrix(mat_gene)))^(1/2)
#   mat_gene = affinityMatrix(mat_gene, K, alpha)


  mat_path = t(mat_path)
  mat_path = standardNormalization(mat_path)

  mat_path = as.matrix(parDist(as.matrix(mat_path), method='euclidean'), threads=400) # 并行的方式去加速计算dist 代表细胞间的correlation
#   mat_path = (dist2(as.matrix(mat_path),as.matrix(mat_path)))^(1/2)
  mat_path = affinityMatrix(mat_path, K, alpha)
#   W = SNF(list(mat_path, mat_gene), K, T)
#   return(W)
  return(mat_path)
}



main<-function(paName, paPath, save_path, mat_gene){
#   original_paName = paName
  print("PaName")
  print(paName)

  gSet = load_pathway(paPath, paste(paName,'.gmt',sep=''))
  gSet = subsetGeneSets(gSet, rownames(mat_gene)) #AUCell

  gSet = clean_sets(gSet) # min.size = 5; max.size = 500

  # pathway scoring: AUCell
  mat_path = pathway_scoring(gSet, mat_gene)

  print("sum of mat_path")
  print(sum(as.matrix(mat_path)))

  return(0)

  # 去掉这一步骤，直接拿到 geneset * cell 的表达矩阵
  W=integrating_pathway(mat_gene, mat_path)
  print("Save the W (integrated) matrix")
  write.csv(W, save_path)
}




# base_path = 'E:\\YuAnHuang\\kevislin\\Cell_Classification\\experiment\\species_v3'
# proj = 'mca_gse84133'
paPath = "E:\\yuanhuang\\kevislin\\data\\pathway\\new_human"
mouse_paname = c('KEGG', 'Reactome', 'Wikipathways', 'biase')
paName = c('KEGG', 'Reactome', 'Wikipathways', 'yan', 'inoh', 'pathbank')

# paName = mouse_paname
args = commandArgs(trailingOnly = TRUE)
base_path = args[[1]]

# query
data_path = paste(base_path, 'raw_data', 'query', sep='\\')
# 多少个query
for(i in 1:1) {
    mat_name = paste('data_', i, '.csv', sep='')
    mat_gene = load_matrix_for_GSE(paste(data_path, mat_name, sep='\\'))

    mat_gene = t(mat_gene) # 对于(cell*genes)格式的数据，先做一次转置
    for(j in 1:4) {
        main(paName[i], paPath, paste(data_path, paste(paste('sm_', i, '_', sep=''), j, '.csv',sep=''), sep='\\'), mat_gene)
    }
}
# 这个old版本有个很奇怪的问题，同一个pathway跑出来的sm的结果会不一样, 我猜测是和并行有关，但是新版的和旧版感觉没差别啊，为啥新版的可以完全相等