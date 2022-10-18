library(conos)
library(pagoda2)

main <- function(path, ref_key, query_key){

    print(path)
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
    rownames(label1) = colnames(data1)
    rownames(label2) = colnames(data2)

#     rownames(data1) = rownames(data2)

    colnames(label1) = c('cell_type1')
    colnames(label2) = c('type')
    label1 = t(label1)

    label1 = setNames(label1, colnames(data1))

    panel <- list(ref=as.matrix(data1), query=as.matrix(data2))
    panel.preprocessed <- lapply(panel, basicP2proc, n.cores=1, min.cells.per.gene=0, n.odgenes=2e3, get.largevis=FALSE, make.geneknn=FALSE)

    con <- Conos$new(panel.preprocessed, n.cores=1)
    con$buildGraph(k=5, k.self=5, k.self.weigh=0.01, ncomps=30, n.odgenes=5e3, space='PCA')
#     con$plotPanel(clustering="multilevel", use.local.clusters=TRUE, title.size=6)
    con$plotPanel(groups = label1)


    pred <- con$propagateLabels(labels = label1, verbose=TRUE)$labels
    pred <- as.matrix(pred)
    pred <- pred[c((dim(data1)[2]+1):dim(pred)[1]),]


    match <- (pred==label2)
    acc = sum(match)/length(match)
    return (acc)
}
final_acc <- c()
path = '../experiment/platform/81608_84133/data'
acc = main(path, '1', '1')
final_acc<-append(final_acc, acc)
print(final_acc)
# acc = main(path, '1', '2')
# final_acc<-append(final_acc, acc)
# acc = main(path, '1', '3')
# final_acc<-append(final_acc, acc)
# acc = main(path, '1', '4')
# final_acc<-append(final_acc, acc)
# acc = main(path, '1', '5')
# final_acc<-append(final_acc, acc)
# # path = '../experiment/platform/85241_84133/data'
# # acc = main(path, '1', '1')
# # final_acc<-append(final_acc, acc)
#
# print(final_acc)
# [1] 0.2481156 0.2040976 0.2232533 0.2243346 0.2448541