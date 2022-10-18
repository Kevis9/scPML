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
  print("xxxx")
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

scName = 'yan'
paPath = "E:\\yuanhuang\\kevislin\\data\\pathway"
# ref 1
data_path = 'E:\\YuAnHuang\\kevislin\\Cell_Classification\\experiment\\species\\gse_emtab_common_type\\mouse_human\\data\\ref'
mat_name = 'data_1.csv'
mat_gene = load_matrix_for_GSE(paste(data_path, mat_name, sep='\\'))
mat_gene = t(mat_gene) # 对于(cell*genes)格式的数据，先做一次转置
main('KEGG', scName,'human', paPath, paste(data_path, 'sm_1_1.csv', sep='\\'))
main('Reactome', scName,'human', paPath, paste(data_path, 'sm_1_2.csv', sep='\\'))
main('Wikipathways', scName,'human', paPath, paste(data_path, 'sm_1_3.csv', sep='\\'))
main('de novo pathway', scName,'human', paPath, paste(data_path, 'sm_1_4.csv', sep='\\'))

# query_1
data_path = 'E:\\YuAnHuang\\kevislin\\Cell_Classification\\experiment\\species\\gse_emtab_common_type\\mouse_human\\data\\query'
mat_name = 'data_1.csv'
mat_gene = load_matrix_for_GSE(paste(data_path, mat_name, sep='\\'))
mat_gene = t(mat_gene) # 对于(cell*genes)格式的数据，先做一次转置
print(dim(mat_gene))
main('KEGG', scName,'human', paPath, paste(data_path, 'sm_1_1.csv', sep='\\'))
main('Reactome', scName,'human', paPath, paste(data_path, 'sm_1_2.csv', sep='\\'))
main('Wikipathways', scName,'human', paPath, paste(data_path, 'sm_1_3.csv', sep='\\'))
main('de novo pathway', scName,'human', paPath, paste(data_path, 'sm_1_4.csv', sep='\\'))

# second
# ref 1
# data_path = 'E:\\YuAnHuang\\kevislin\\Cell_Classification\\experiment\\platform\\84133_81608\\data\\ref'
# mat_name = 'data_1.csv'
# mat_gene = load_matrix_for_GSE(paste(data_path, mat_name, sep='\\'))
# mat_gene = t(mat_gene) # 对于(cell*genes)格式的数据，先做一次转置
# main('KEGG', scName,'human', paPath, paste(data_path, 'sm_1_1.csv', sep='\\'))
# main('Reactome', scName,'human', paPath, paste(data_path, 'sm_1_2.csv', sep='\\'))
# main('Wikipathways', scName,'human', paPath, paste(data_path, 'sm_1_3.csv', sep='\\'))
# main('de novo pathway', scName,'human', paPath, paste(data_path, 'sm_1_4.csv', sep='\\'))
#
# # query_1
# data_path = 'E:\\YuAnHuang\\kevislin\\Cell_Classification\\experiment\\platform\\84133_81608\\data\\query'
# mat_name = 'data_1.csv'
# mat_gene = load_matrix_for_GSE(paste(data_path, mat_name, sep='\\'))
# mat_gene = t(mat_gene) # 对于(cell*genes)格式的数据，先做一次转置
# print(dim(mat_gene))
# main('KEGG', scName,'human', paPath, paste(data_path, 'sm_1_1.csv', sep='\\'))
# main('Reactome', scName,'human', paPath, paste(data_path, 'sm_1_2.csv', sep='\\'))
# main('Wikipathways', scName,'human', paPath, paste(data_path, 'sm_1_3.csv', sep='\\'))
# main('de novo pathway', scName,'human', paPath, paste(data_path, 'sm_1_4.csv', sep='\\'))
#
# #third
# # ref 1
# data_path = 'E:\\YuAnHuang\\kevislin\\Cell_Classification\\experiment\\platform\\81608_84133\\data\\ref'
# mat_name = 'data_1.csv'
# mat_gene = load_matrix_for_GSE(paste(data_path, mat_name, sep='\\'))
# mat_gene = t(mat_gene) # 对于(cell*genes)格式的数据，先做一次转置
# main('KEGG', scName,'human', paPath, paste(data_path, 'sm_1_1.csv', sep='\\'))
# main('Reactome', scName,'human', paPath, paste(data_path, 'sm_1_2.csv', sep='\\'))
# main('Wikipathways', scName,'human', paPath, paste(data_path, 'sm_1_3.csv', sep='\\'))
# main('de novo pathway', scName,'human', paPath, paste(data_path, 'sm_1_4.csv', sep='\\'))
#
# # query_1
# data_path = 'E:\\YuAnHuang\\kevislin\\Cell_Classification\\experiment\\platform\\81608_84133\\data\\query'
# mat_name = 'data_1.csv'
# mat_gene = load_matrix_for_GSE(paste(data_path, mat_name, sep='\\'))
# mat_gene = t(mat_gene) # 对于(cell*genes)格式的数据，先做一次转置
# print(dim(mat_gene))
# main('KEGG', scName,'human', paPath, paste(data_path, 'sm_1_1.csv', sep='\\'))
# main('Reactome', scName,'human', paPath, paste(data_path, 'sm_1_2.csv', sep='\\'))
# main('Wikipathways', scName,'human', paPath, paste(data_path, 'sm_1_3.csv', sep='\\'))
# main('de novo pathway', scName,'human', paPath, paste(data_path, 'sm_1_4.csv', sep='\\'))

# mat_name = 'data_2.csv'
# mat_gene = load_matrix_for_GSE(paste(data_path, mat_name, sep='\\'))
# mat_gene = t(mat_gene) # 对于(cell*genes)格式的数据，先做一次转置
# main('KEGG', scName,'human', paPath, paste(data_path, 'sm_2_1.csv', sep='\\'))
# main('Reactome', scName,'human', paPath, paste(data_path, 'sm_2_2.csv', sep='\\'))
# main('Wikipathways', scName,'human', paPath, paste(data_path, 'sm_2_3.csv', sep='\\'))
# main('de novo pathway', scName,'human', paPath, paste(data_path, 'sm_2_4.csv', sep='\\'))
#
# mat_name = 'data_3.csv'
# mat_gene = load_matrix_for_GSE(paste(data_path, mat_name, sep='\\'))
# mat_gene = t(mat_gene) # 对于(cell*genes)格式的数据，先做一次转置
# main('KEGG', scName,'human', paPath, paste(data_path, 'sm_3_1.csv', sep='\\'))
# main('Reactome', scName,'human', paPath, paste(data_path, 'sm_3_2.csv', sep='\\'))
# main('Wikipathways', scName,'human', paPath, paste(data_path, 'sm_3_3.csv', sep='\\'))
# main('de novo pathway', scName,'human', paPath, paste(data_path, 'sm_3_4.csv', sep='\\'))
#
# mat_name = 'data_4.csv'
# mat_gene = load_matrix_for_GSE(paste(data_path, mat_name, sep='\\'))
# mat_gene = t(mat_gene) # 对于(cell*genes)格式的数据，先做一次转置
# main('KEGG', scName,'human', paPath, paste(data_path, 'sm_4_1.csv', sep='\\'))
# main('Reactome', scName,'human', paPath, paste(data_path, 'sm_4_2.csv', sep='\\'))
# main('Wikipathways', scName,'human', paPath, paste(data_path, 'sm_4_3.csv', sep='\\'))
# main('de novo pathway', scName,'human', paPath, paste(data_path, 'sm_4_4.csv', sep='\\'))
#
# mat_name = 'data_5.csv'
# mat_gene = load_matrix_for_GSE(paste(data_path, mat_name, sep='\\'))
# mat_gene = t(mat_gene) # 对于(cell*genes)格式的数据，先做一次转置
# main('KEGG', scName,'human', paPath, paste(data_path, 'sm_5_1.csv', sep='\\'))
# main('Reactome', scName,'human', paPath, paste(data_path, 'sm_5_2.csv', sep='\\'))
# main('Wikipathways', scName,'human', paPath, paste(data_path, 'sm_5_3.csv', sep='\\'))
# main('de novo pathway', scName,'human', paPath, paste(data_path, 'sm_5_4.csv', sep='\\'))
#





