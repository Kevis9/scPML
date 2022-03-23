# install.packages("SNFtool",repos="https://mirrors.tuna.tsinghua.edu.cn/CRAN/")
install.packages("BiocManager",repos="https://mirrors.tuna.tsinghua.edu.cn/CRAN/")
# BiocManager::install("AUCell")
# BiocManager::install("SingleCellExperiment")
BiocManager::install("GSEABase")
# install.packages("AUCell")
# install.packages("SingleCellExperiment",repos="https://mirrors.tuna.tsinghua.edu.cn/CRAN/")
library(SNFtool) # SNF;spectralClustering
library(GSEABase) # getGmt, load pathway information
library(AUCell) # AUCell, pathway scoring method 
# library(SingleCellExperiment)

# clustering method
# library(stats) # kmeans
# library(fastcluster) # fastcluster::hclust
# library(dbscan) # dbscan
# library(wordspace) # dist.matrix, fast distance calculation function
# library(SC3) # SC3
# library(Seurat)# Seurat
# library(cidr) # CIDR
# library(pcaReduce) # pcaReduce
# library(SOUP) # SOUP
# source("..//SOUP_ori.R") 
# library(reticulate) # for python SNN-Cliq
# py_config() # config python
# source_python("..//SNN-Cliq.py") # SNN-Cliq

# load cell label
load_label <- function(path,name){
  label = read.table(paste(path,name,sep='/'),sep='\t')
  return(as.vector(label$'cell_type1'))
}

# load single cell expresion matrix
# 原来的做法
load_matrix <- function(path, name){
  expr_matrix = read.table(paste(path,name,sep='/'),sep='\t')  
  expr_matrix = as.matrix(expr_matrix)
  
  return(expr_matrix)
}

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

name_2_species<-function(scName){
  if(scName == 'yan')
  {return('human')}
  else{return('mouse')} # scName=='biase'
}

# create de novo pathway
# hierarchical clustering from hclut pearson, dendrogram, cluster=150(PAGODA)
create_denovo_pathway <- function(mat_gene){
  # gene clutering
  # see pagoda.gene.clusters for details
  n.clusters = 150
  n.cores = 4
  cor.method = 'p'
  method = 'ward.D'
  vi <- which(abs(apply(mat_gene, 1, function(x) sum(abs(diff(x)))))>0)
  gd <- as.dist(1 - WGCNA::cor(t(mat_gene)[, vi], method = cor.method, nThreads = n.cores))
  gcl <- fastcluster::hclust(gd, method = method)
  gcll <- cutree(gcl, n.clusters)
  gcls <- tapply(rownames(mat_gene)[vi], as.factor(gcll), I)
  
  # create gene_sets
  geneIdType = NullIdentifier()
  collectionType = NullCollection()
  template <- GeneSet(geneIdType = geneIdType, collectionType = collectionType)
  gene_set = 
    GeneSetCollection(lapply(seq(1:length(gcls)), function(i) {
      initialize(template, geneIds = as.vector(gcls[[i]]), 
                 setName = as.character(i), shortDescription = as.character(i),
                 setIdentifier = template@setIdentifier)
    }))
  return(gene_set)
}

# pathway scoring: AUCell
pathway_scoring <- function(gSet, mat_gene){
  cells_rankings <- AUCell_buildRankings(mat_gene, nCores=1, plotStats=TRUE)
  cells_AUC <- AUCell_calcAUC(gSet, cells_rankings, nCores=1,aucMaxRank = ceiling(0.05 * nrow(cells_rankings)))
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

  mat_gene = t(mat_gene)
  mat_gene = standardNormalization(mat_gene)
  mat_gene = (dist2(as.matrix(mat_gene),as.matrix(mat_gene)))^(1/2)
  mat_gene = affinityMatrix(mat_gene, K, alpha)
  
  mat_path = t(mat_path)
  mat_path = standardNormalization(mat_path)
  mat_path = (dist2(as.matrix(mat_path),as.matrix(mat_path)))^(1/2)
  mat_path = affinityMatrix(mat_path, K, alpha)
  W = SNF(list(mat_path, mat_gene), K, T)
  return(W)
}


clustering_by_integrating_pathway<- function(mat_gene,mat_path,W,cName,k){
  dis_W = dist.matrix(W,method = "euclidean",as.dist = TRUE)
  m_list = c('kmeans','hierarchical','spectral','DBSCAN',
             'SC3','Seurat','CIDR','pcaReduce','SOUP','SNN-Cliq')
  switch(which(m_list == cName),
         # 1.keams, input:dist
         {c = kmeans(x=as.dist(dis_W),centers=k); 
         clust_results =  c$cluster},
         
         # 2.hierarchical, input:dist
         {gcl <- fastcluster::hclust(as.dist(dis_W), method = 'ward.D');
         clust_results <- cutree(gcl, k)},
         
         # 3.spectral, input:sim
         {K = ceiling(dim(mat_gene)[2]/10)
         sim_gene = affinityMatrix(as.matrix(dis_W), K, 0.5);
         clust_results = spectralClustering(sim_gene, k)},
         
         # 4.DBSCAN, input:dis
         {b = sort(kNNdist(as.dist(W),5)); # dbscan::kNNdistplot(as.dist(1/w_gene), k =  5)
         eps = b[ceiling(0.90*length(b))];
         d = dbscan(x=as.dist(W),eps);
         clust_results = d$cluster},
         
         # 5.SC3
         {sce <- SingleCellExperiment(assays = list(counts = mat_gene,logcounts = mat_gene),colData = colnames(mat_gene))
         rowData(sce)$feature_symbol <- rownames(sce)
         a <- sc3(sce, ks = k, biology = FALSE, gene_filter = FALSE) #SC3
         concens_mat = a@metadata$sc3$consensus[[as.character(k)]]$consensus   # concens matrix
         W=adding_pathway(concens_mat, mat_path)
         y = spectralClustering(W, k)
         clust_results= as.numeric(y)},
         
         # 6. Seurat
         {pbmc <- CreateSeuratObject(counts=mat_gene)
         pbmc <- FindVariableFeatures(object = pbmc)
         pbmc[["percent.mt"]] <- PercentageFeatureSet(object = pbmc, pattern = "^mt-")
         pbmc <- ScaleData(object = pbmc, vars.to.regress = "percent.mt")
         pbmc <- RunPCA(object = pbmc)
         a = pbmc@reductions$pca@cell.embeddings
         b = a[,1:10]
         W=adding_pathway(t(b), mat_path)
         y = spectralClustering(W, k)
         clust_results= as.numeric(y)},
         
         # 7. CIDR
         {sData <- new("scData", tags = W, tagType = 'CPM')
         sData@dissim <- as.matrix(dist.matrix(t(W),method = "euclidean",as.dist = TRUE))
         sData <- scPCA(sData,plotPC = FALSE)
         sData <- nPC(sData)
         sData <- scCluster(sData,nCluster =k)
         clust_results= as.numeric(sData@clusters)},
         
         # 8. pcaReduce
         {Output_S <- PCAreduce(W, nbt=1, q=k-1, method='S')
         clust_results = Output_S[[1]][,1]},
         
         # 9. SOUP
         {soup.out = SOUP_2(W, Ks=k, type="log")
         soup.labels = soup.out$major.labels[[1]]
         clust_results = as.numeric(soup.labels)},
		 
		 # 10. SNN-Cliq
		 {clust_results = snn_cliq(as.matrix(dis_W), 0.5, 0.7)}
  )
  return(clust_results)
}


# main function
# cName: Name of clustering method
#     'kmeans','hierarchical','spectral','DBSCAN',
#     'SC3','Seurat','CIDR','pcaReduce','SOUP','SNN-Cliq'
# paName: Name of pathway database
#     KEGG, Wikipathways, Reactome, de novo pathway
# scName: Name of singel cell dataset
#     'yan', 'biase'
# labelPath = '../Demo_data/label'
# scPath = '../Demo_data/matrix'


main<-function(paName, scName,s, paPath, scPath, save_path){
  print("start")
  demoDatas = c('yan','biase')
  # load singel cell data
  
  # mat_gene = load_matrix_for_GSE('/Users/kevislin/Desktop/单细胞/资料汇总/data/transfer_across_species_data/scData/mouse_pancreas.csv')  
  # mat_gene = load_matrix_for_GSE('/Users/kevislin/Desktop/test.csv')  
  # mat_gene = load_matrix(scPath, scName) 
  
  # mat_gene = t(mat_gene) # 对于(cell*genes)格式的数据，先做一次转置 
  original_paName = paName
  # load pathway
  if(s==''){
    s = name_2_species(scName)
  }
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
  
  # load cell label
  # label = load_label(labelPath, paste(scName,'label',sep='_'))
  # label_int = as.numeric(as.vector(factor(label,levels=unique(label),labels=seq(1:length(unique(label))))))
  
  # integrating pathway 
  W=integrating_pathway(mat_gene, mat_path)

  print("Save the W (integrated) matrix")
  filepath = paste(save_path, original_paName, '.csv',sep='')
  write.table(W, file=filepath, sep=',', row.names=TRUE, col.names=TRUE,quote=FALSE)
  # write.table(W, file='./W.csv', sep=',', row.names=TRUE, col.names=TRUE,quote=FALSE)
  # write.table(mat_gene, file='./mat_gene.csv', sep=',', row.names=TRUE, col.names=TRUE,quote=FALSE)
  # write.table(label_int, file='./label.csv', sep=',', row.names=TRUE, col.names=TRUE,quote=FALSE)
  # clustering
  # k = length(unique(label_int)) # real k
  # clust_results = clustering_by_integrating_pathway(mat_gene,mat_path,W,cName,k)
  # print(clust_results)
}



scName= 'yan'
paPath = "/home/yuanhuang/kevislin/data/pathway"
mat_path = '/home/yuanhuang/kevislin/data/transfer_across_platforms/PBMC/cel_seq_10x_v3/'

mat_name = 'cel_seq2_data.csv'
mat_gene = load_matrix_for_GSE(paste(mat_path, mat_name, sep=''))
mat_gene = t(mat_gene) # 对于(cell*genes)格式的数据，先做一次转置
save_path = paste(mat_path, 'similarity_mat/SM_cel_seq_', sep='')
main('KEGG', scName,'human', paPath, save_path)
main('Reactome', scName,'human', paPath, save_path)
main('Wikipathways', scName,'human', paPath, save_path)
main('de novo pathway', scName,'human', paPath, save_path)


mat_name = '10x_v3_data.csv'
mat_gene = load_matrix_for_GSE(paste(mat_path, mat_name, sep=''))
mat_gene = t(mat_gene) # 对于(cell*genes)格式的数据，先做一次转置
save_path = paste(mat_path, 'similarity_mat/SM_10x_v3_', sep='')
main('KEGG', scName,'human', paPath, save_path)
main('Reactome', scName,'human', paPath, save_path)
main('Wikipathways', scName,'human', paPath, save_path)
main('de novo pathway', scName,'human', paPath, save_path)


