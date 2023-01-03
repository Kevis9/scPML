library(conos)
library(pagoda2)
library(Seurat)

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

    label1 = as.matrix(read.csv(paste(path, 'ref', ref_label, sep='/'))
    cellannot = data.frame(
        cell_name = colnames(data1),
        cell_type = label1[, 1]
    )
    cellannot = setNames(cellannot[, 2], cellannot[, 1])

    label2 = read.csv(paste(path, 'query', query_label, sep='/'))

    panel <- list(ref=as.matrix(data1), query=as.matrix(data2))
    panel.preprocessed <- lapply(panel, basicSeuratProc)
    con <- Conos$new(panel.preprocessed, n.cores=1)
    con$buildGraph(k=30, k.self=5, space='PCA', ncomps=30, n.odgenes=2000, matching.method='mNN', metric='angular', score.component.variance=TRUE, verbose=TRUE)

    #     con$plotPanel(clustering="multilevel", use.local.clusters=TRUE, title.size=6)
#     con$plotPanel(groups = label1)

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