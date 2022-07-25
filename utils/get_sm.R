# if (!require("BiocManager", quietly = TRUE))
#     install.packages("BiocManager", repos="https://mirrors.tuna.tsinghua.edu.cn/CRAN/")
# BiocManager::install("SNFtool")
# BiocManager::install("GSEABase")
# BiocManager::install("AUCell")
library(parallelDist)
library(SNFtool) # SNF;spectralClustering
library(GSEABase) # getGmt, load pathway information
library(AUCell) # AUCell, pathway scoring method 



# 为GSE84133_RAW设置的load_matrix
load_matrix_for_GSE <- function(path){
  expr_matrix = read.csv(path,check.names=FALSE) # 去掉X  
  # print(colnames(expr_matrix))  
  # print(rownames(expr_matrix))
  expr_matrix = expr_matrix[,-1]#删掉第1列 cell id
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
  cells_rankings <- AUCell_buildRankings(mat_gene, nCores=25, plotStats=TRUE)
  cells_AUC <- AUCell_calcAUC(gSet, cells_rankings, nCores=25, aucMaxRank = ceiling(0.05 * nrow(cells_rankings)))
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



main<-function(paName, scName,s, paPath, save_path){

  demoDatas = c('yan','biase')

  original_paName = paName
  # load pathway
  if(paName=='de novo pathway'){
    if(scName %in% demoDatas){
      paName=paste(scName,'150.gmt',sep='_')
      gSet = load_pathway(paPath,paName)}
    else{
      gSet = create_denovo_pathway(mat_gene)}}
  else{
    paName=paste(paName,'_',s,'.gmt',sep='')
    gSet = load_pathway(paPath,paName)
  }

  gSet = subsetGeneSets(gSet, rownames(mat_gene)) #AUCell
  gSet = clean_sets(gSet) # min.size = 5; max.size = 500

  # pathway scoring: AUCell
  mat_path = pathway_scoring(gSet, mat_gene)

  W=integrating_pathway(mat_gene, mat_path)


  print("Save the W (integrated) matrix")

  if(original_paName=='de novo pathway'){
       if(s=='human'){
        original_paName='yan'
       } else{
        original_paName = 'biase'
       }
  }
#   filepath = paste(save_path, original_paName, '.csv',sep='')

  write.table(W, file=save_path, sep=',', row.names=TRUE, col.names=TRUE,quote=FALSE)

}

scName= 'yan'
paPath = "F:\\yuanhuang\\kevislin\\data\\pathway"
# ref
data_path = 'F:\\yuanhuang\\kevislin\\Cell_Classification\\experiment3\\platform\\task6\\data\\ref'
mat_name = 'data_1.csv'
mat_gene = load_matrix_for_GSE(paste(data_path, mat_name, sep='\\'))
mat_gene = t(mat_gene) # 对于(cell*genes)格式的数据，先做一次转置
main('KEGG', scName,'human', paPath, paste(data_path, 'sm_1_1.csv', sep='\\'))
main('Reactome', scName,'human', paPath, paste(data_path, 'sm_1_2.csv', sep='\\'))
main('Wikipathways', scName,'human', paPath, paste(data_path, 'sm_1_3.csv', sep='\\'))
main('de novo pathway', scName,'human', paPath, paste(data_path, 'sm_1_4.csv', sep='\\'))
q()
# query_1
# mat_name = 'mouse_data.csv'
# scName= 'biase'
data_path = 'F:\\yuanhuang\\kevislin\\Cell_Classification\\experiment3\\platform\\task6\\data\\query'
mat_name = 'data_1.csv'
mat_gene = load_matrix_for_GSE(paste(data_path, mat_name, sep='\\'))
mat_gene = t(mat_gene) # 对于(cell*genes)格式的数据，先做一次转置
main('KEGG', scName,'human', paPath, paste(data_path, 'sm_1_1.csv', sep='\\'))
main('Reactome', scName,'human', paPath, paste(data_path, 'sm_1_2.csv', sep='\\'))
main('Wikipathways', scName,'human', paPath, paste(data_path, 'sm_1_3.csv', sep='\\'))
main('de novo pathway', scName,'human', paPath, paste(data_path, 'sm_1_4.csv', sep='\\'))

q()
data_path = 'F:\\yuanhuang\\kevislin\\Cell_Classification\\experiment\\platform\\task5\\data\\ref'
mat_gene = load_matrix_for_GSE(paste(data_path, mat_name, sep='\\'))
mat_gene = t(mat_gene) # 对于(cell*genes)格式的数据，先做一次转置
main('KEGG', scName,'human', paPath, paste(data_path, 'sm_1_1.csv', sep='\\'))
main('Reactome', scName,'human', paPath, paste(data_path, 'sm_1_2.csv', sep='\\'))
main('Wikipathways', scName,'human', paPath, paste(data_path, 'sm_1_3.csv', sep='\\'))
main('de novo pathway', scName,'human', paPath, paste(data_path, 'sm_1_4.csv', sep='\\'))

data_path = 'F:\\yuanhuang\\kevislin\\Cell_Classification\\experiment\\platform\\task6\\data\\query'
mat_gene = load_matrix_for_GSE(paste(data_path, mat_name, sep='\\'))
mat_gene = t(mat_gene) # 对于(cell*genes)格式的数据，先做一次转置
main('KEGG', scName,'human', paPath, paste(data_path, 'sm_1_1.csv', sep='\\'))
main('Reactome', scName,'human', paPath, paste(data_path, 'sm_1_2.csv', sep='\\'))
main('Wikipathways', scName,'human', paPath, paste(data_path, 'sm_1_3.csv', sep='\\'))
main('de novo pathway', scName,'human', paPath, paste(data_path, 'sm_1_4.csv', sep='\\'))
q()


# query_2
# mat_name = 'mouse_data.csv'
# scName= 'biase'
data_path = 'F:\\yuanhuang\\kevislin\\data\\platform\\PBMC_with_all_common_type\\task1\\query\\query_2'

mat_gene = load_matrix_for_GSE(paste(data_path, mat_name, sep='\\'))
mat_gene = t(mat_gene) # 对于(cell*genes)格式的数据，先做一次转置
save_path = paste(data_path, 'sm_', sep='\\')
main('KEGG', scName,'human', paPath, save_path)
main('Reactome', scName,'human', paPath, save_path)
main('Wikipathways', scName,'human', paPath, save_path)
main('de novo pathway', scName,'human', paPath, save_path)

#query_3
data_path = 'F:\\yuanhuang\\kevislin\\data\\platform\\PBMC_with_all_common_type\\task1\\query\\query_3'

mat_gene = load_matrix_for_GSE(paste(data_path, mat_name, sep='\\'))
mat_gene = t(mat_gene) # 对于(cell*genes)格式的数据，先做一次转置
save_path = paste(data_path, 'similarity_mat/SM_', sep='\\')
main('KEGG', scName,'human', paPath, save_path)
main('Reactome', scName,'human', paPath, save_path)
main('Wikipathways', scName,'human', paPath, save_path)
main('de novo pathway', scName,'human', paPath, save_path)

# query_4
data_path = 'F:\\yuanhuang\\kevislin\\data\\platform\\PBMC_with_all_common_type\\task1\\query\\query_4'

mat_gene = load_matrix_for_GSE(paste(data_path, mat_name, sep='\\'))
mat_gene = t(mat_gene) # 对于(cell*genes)格式的数据，先做一次转置
save_path = paste(data_path, 'similarity_mat/SM_', sep='\\')
main('KEGG', scName,'human', paPath, save_path)
main('Reactome', scName,'human', paPath, save_path)
main('Wikipathways', scName,'human', paPath, save_path)
main('de novo pathway', scName,'human', paPath, save_path)

#query_5

data_path = 'F:\\yuanhuang\\kevislin\\data\\platform\\PBMC_with_all_common_type\\task1\\query\\query_5'

mat_gene = load_matrix_for_GSE(paste(data_path, mat_name, sep='\\'))
mat_gene = t(mat_gene) # 对于(cell*genes)格式的数据，先做一次转置
save_path = paste(data_path, 'similarity_mat/SM_', sep='\\')
main('KEGG', scName,'human', paPath, save_path)
main('Reactome', scName,'human', paPath, save_path)
main('Wikipathways', scName,'human', paPath, save_path)
main('de novo pathway', scName,'human', paPath, save_path)


# main('pid', scName,'human', paPath, save_path)
# main('inoh', scName,'human', paPath, save_path)
# main('humancyc', scName,'human', paPath, save_path)
# main('panther', scName,'human', paPath, save_path)


#
# scName = 'yan'
# # human
# mat_name = 'human_data.csv'
# mat_gene = load_matrix_for_GSE(paste(data_path, mat_name, sep='/'))
# mat_gene = t(mat_gene) # 对于(cell*genes)格式的数据，先做一次转置
# save_path = paste(data_path, 'similarity_mat/SM_human_', sep='/')
# main('KEGG', scName,'human', paPath, save_path)
# main('Reactome', scName,'human', paPath, save_path)
# main('Wikipathways', scName,'human', paPath, save_path)
# main('de novo pathway', scName,'human', paPath, save_path)

# mouse_human
# data_path = '/home/zhianhuang/yuanhuang/kevislin/data/species_data/GSE84133/mouse_human'
# mat_name = 'human_data.csv'
# mat_gene = load_matrix_for_GSE(paste(data_path, mat_name, sep='/'))
# mat_gene = t(mat_gene) # 对于(cell*genes)格式的数据，先做一次转置
# save_path = paste(data_path, 'similarity_mat2/SM_human_', sep='/')
# main('KEGG', scName,'human', paPath, save_path)
# main('Reactome', scName,'human', paPath, save_path)
# main('Wikipathways', scName,'human', paPath, save_path)
# main('de novo pathway', scName,'human', paPath, save_path)
