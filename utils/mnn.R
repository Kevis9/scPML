# 这部分暂时建议直接手动
library(batchelor)
library(umap)

ref_data = read.csv('../experiment/omic/kidney/ref_norm_data.csv', header=FALSE)
query_data = read.csv('../experiment/omic/kidney/query_norm_data.csv', header=FALSE)
data = rbind(ref_data, query_data)
labels = c()
for(i in c(1:dim(ref_data)[1])){
    labels = append(labels, "red")
}
for(i in c(1:dim(query_data)[1])){
    labels = append(labels, "blue")
}

# data.umap = umap(data)
# plot(data.umap$layout, col=labels, pch=16, asp = 1)

out = mnnCorrect(t(ref_data), t(query_data))

new_data = out@assays@data@listData$corrected
# new_data.umap = umap(t(new_data))
# plot(new_data.umap$layout, col=labels, pch=16, asp = 1)

new_ref_data = t(new_data[, 1:dim(ref_data)[1]])
new_query_data = t(new_data[, (dim(ref_data)[1]+1):dim(new_data)[2]])

write.csv(new_ref_data, 'new_ref_data.csv')
write.csv(new_query_data, 'new_query_data.csv')

###
# new_ref_data = read.csv('new_ref_data.csv', header=TRUE, row.names=1)
# new_query_data = read.csv('new_query_data.csv', header=TRUE, row.names=1)
# data = rbind(new_ref_data, new_query_data)