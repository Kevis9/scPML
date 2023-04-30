# simulation, randomly set 0
# p: the proprotion of random selection, c(0.05,0.10,0.15,0.20)
# t: simulation times, c(1,2,3)
simulation_0 <- function(mat_gene,p,t){
  p = as.numeric(p)
  t = as.numeric(t)
  set.seed(p*100+t)
  index = matrix(sample(c(TRUE,FALSE), length(mat_gene), p = c(1,1-p), replace=TRUE), nrow=nrow(mat_gene))
  newMat = mat_gene
  colnames(newMat) = colnames(mat_gene)
  rownames(newMat) = rownames(mat_gene)
#   rm(mat_gene,gMean,gSd);gc()
  newMat[index==TRUE]=0
  return(newMat)
}

# simulation, gaussian
# p: noise proprotion, c(0.05,0.10,0.15,0.20)
# t: simulation times, c(1,2,3)
simulation_gaussian <- function(mat_gene,p,t){
  p = as.numeric(p)
  t = as.numeric(t)
  set.seed(p*100+t)
  cNum = length(colnames(mat_gene))
  gMean = apply(mat_gene,1,mean)
  gSd = apply(mat_gene,1,sd)
  noise = t(sapply(1:length(gMean),function(i) rnorm(cNum,mean=gMean[i],sd=gSd[i])))
  newMat = mat_gene*(1-p) + noise*(p)
  colnames(newMat) = colnames(mat_gene)
  rownames(newMat) = rownames(mat_gene)
#   rm(mat_gene,gMean,gSd);rm(noise);gc()
  newMat[newMat < 0] <- 0
  return(newMat)
}

read_data <- function(path) {
    # return matrix
    data = as.matrix(read.csv(path, row.names=1))
    return (data)
}

path = 'raw_data/ref/data_1.csv'
mat_gene = read_data(path)
random_percent = c(0.05, 0.10, 0.15, 0.2)
for (p in random_percent){
   newMat_0 = simulation_0(mat_gene, p, 1)
   newMat_gaussian = simulation_gaussian(mat_gene, p, 1)
   save_path = paste('dropout', p , sep='/')
   if(!file.exists(save_path)) {
        dir.create(save_path)
        dir.create(paste(save_path, 'raw_data', sep='/'))
        dir.create(paste(save_path, 'raw_data', 'ref', sep='/'))
        dir.create(paste(save_path, 'raw_data', 'query', sep='/'))

        dir.create(paste(save_path, 'data', sep='/'))
        dir.create(paste(save_path, 'data', 'ref', sep='/'))
        dir.create(paste(save_path, 'data', 'query', sep='/'))

   }
   save_path = paste(save_path, 'raw_data', 'ref', 'data_1.csv',sep='/')
   write.csv(newMat_0, save_path)

   save_path = paste('gaussian', p , sep='/')
   if(!file.exists(save_path)) {
        dir.create(save_path)
        dir.create(paste(save_path, 'raw_data', sep='/'))
        dir.create(paste(save_path, 'raw_data', 'ref', sep='/'))
        dir.create(paste(save_path, 'raw_data', 'query', sep='/'))

        dir.create(paste(save_path, 'data', sep='/'))
        dir.create(paste(save_path, 'data', 'ref', sep='/'))
        dir.create(paste(save_path, 'data', 'query', sep='/'))

   }
   save_path = paste(save_path, 'raw_data', 'ref', 'data_1.csv', sep='/')
   write.csv(newMat_gaussian, file=save_path)
}
