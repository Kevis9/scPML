library(batchelor)
library(umap)
ref_data = read.csv('ref_data.csv', header=FALSE)
query_data = read.csv('query_data.csv', header=FALSE)
data = rbind(ref_data, query_data)
labels = c()
for(i in c(1:dim(ref_data)[1])){
  labels = append(labels, "red")
}
for(i in c(1:dim(query_data)[1])){
  labels = append(labels, "blue")
}
data.umap = umap(data)
plot(data.umap, col=labels, pch=16, asp = 1)

out = mnnCorrect(t(ref_data), t(query_data))