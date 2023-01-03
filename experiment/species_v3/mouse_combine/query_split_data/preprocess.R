library(batchelor)
library(Seurat)

read_data <- function(path) {
    # return matrix
    data = as.matrix(read.csv(path, row.names=1))
    return (data)
}

read_label <- function(path) {
    #return matrix
    label = as.matrix(read.csv(path))
    return (label)
}

select_feature <- function(data,label,nf=2000){
    M <- nrow(data); new.label <- label[,1]
    pv1 <- sapply(1:M, function(i){
        mydataframe <- data.frame(y=as.numeric(data[i,]), ig=new.label)
        fit <- aov(y ~ ig, data=mydataframe)
        summary(fit)[[1]][["Pr(>F)"]][1]})
    names(pv1) <- rownames(data)
    pv1.sig <- names(pv1)[order(pv1)[1:nf]]
    egen <- unique(pv1.sig)
    return (egen)
}

normalize_data <- function(data) {
#     data <- as.matrix(Seurat:::NormalizeData.default(data,verbose=F))
    data = t(data) # 先转置，变成cell * gene
    row_sum = apply(data, 1, sum)
    mean_t = mean(row_sum)
#     细胞表达量为0的地方不用管，设置为1，表示不影响
    row_sum[row_sum==0] = 1
#
#     row_sum是vector，会自动广播
    data = data/row_sum * mean_t
#     再次转置回来，变成 gene * cell
    data = t(data)
    return (data)

}



main <- function(){

    ref_data = t(read_data("../raw_data/ref/data_1.csv")) # gene x cell
    ref_label = read_label("../raw_data/ref/label_1.csv")
    query_data_1 = t(read_data('emtab_data.csv'))
    query_data_2 =  t(read_data('gse81608_data.csv'))

    query_data_3 =  t(read_data('gse85241_data.csv'))
    query_data_4 =  t(read_data('gsehuman_data.csv'))


    # gene selection
    print(dim(ref_data))
    print(dim(ref_label))
    sel.features <- select_feature(ref_data, ref_label)
    sel.ref_data = ref_data[sel.features, ]
    sel.query_data_1 = query_data_1[sel.features, ]
    sel.query_data_2 = query_data_2[sel.features, ]
    sel.query_data_3 = query_data_3[sel.features, ]
    sel.query_data_4 = query_data_4[sel.features, ]

    # Norm
    norm.ref_data = normalize_data(sel.ref_data)
    norm.query_data_1 = normalize_data(sel.query_data_1)
    norm.query_data_2 = normalize_data(sel.query_data_2)
    norm.query_data_3 = normalize_data(sel.query_data_3)
    norm.query_data_4 = normalize_data(sel.query_data_4)

    # Mnn correct
#     out = mnnCorrect(norm.ref_data, norm.query_data, cos.norm.in = FALSE, cos.norm.out=FALSE)
    out = mnnCorrect(norm.ref_data, norm.query_data_1, norm.query_data_2, norm.query_data_3, norm.query_data_4, sigma=0.3)

    new_data = out@assays@data@listData$corrected

    new.ref_data = t(out@assays@data$corrected[,out$batch==1])
    new.query_data_1 = t(out@assays@data$corrected[,out$batch==2])
    new.query_data_2 = t(out@assays@data$corrected[,out$batch==3])
    new.query_data_3 = t(out@assays@data$corrected[,out$batch==4])
    new.query_data_4 = t(out@assays@data$corrected[,out$batch==5])
#     write.csv(new.ref_data, file=paste(save_path, 'ref', 'data_1.csv', sep='/'), row.names=TRUE)
#     write.csv(new.query_data, file=paste(save_path, 'query', 'data_1.csv', sep='/'), row.names=TRUE)
      write.csv(new.ref_data, file='aligned/mouse_data.csv', row.names=TRUE)
      write.csv(new.query_data_1, file="aligned/emtab_data.csv", row.names=TRUE)
      write.csv(new.query_data_2, file="aligned/gse81608_data.csv", row.names=TRUE)
      write.csv(new.query_data_3, file="aligned/gse85241_data.csv", row.names=TRUE)
      write.csv(new.query_data_4, file="aligned/gsehuman_data.csv", row.names=TRUE)

}


main()


