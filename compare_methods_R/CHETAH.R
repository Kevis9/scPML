library(CHETAH)
print("start")
path = 'experiment\\platform\\task_seq_all\\data'
print(path)
data1 = t(as.data.frame(read.csv(paste(path, 'ref', 'data_1.csv', sep='\\'), row.names=1))) # 变为gene * cell
data2 = t(as.data.frame(read.csv(paste(path, 'query', 'data_5.csv', sep='\\'), row.names=1)))
print(dim(data1))
print(dim(data2))
label1 = as.data.frame(read.csv(paste(path, 'ref', 'label_1.csv', sep='\\')))
rownames(label1) = colnames(data1)
label2 = as.data.frame(read.csv(paste(path, 'query', 'label_5.csv', sep='\\')))
rownames(label2) = colnames(data2)
rownames(data1) = rownames(data2)
colnames(label1) = c('celltypes')
colnames(label2) = c('celltypes')

ref_counts = data1
ref_ct = label1
reference <- SingleCellExperiment(assays = list(counts = ref_counts),
                                     colData = ref_ct)
input <- SingleCellExperiment(assays = list(counts = data2))
input <- CHETAHclassifier(input = input, ref_cells = reference)
## Extract celltypes:
celltypes <- input$celltype_CHETAH
match <- (celltypes==label2)
print(sum(match)/length(match))
