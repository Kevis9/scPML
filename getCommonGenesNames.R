
# 同源基因获取
# 安装homologene这个R包
# install.packages('homologene',repos="https://mirrors.tuna.tsinghua.edu.cn/CRAN/")
#加载homologene这个R包
library(homologene)

# 输出homologene支持的物种号
# homologene::taxData

# 给小鼠的基因
# mouse_gene_list<-c("Acadm","0610009B22Rik","0610007P14Rik")
mouse_gene_df = t(as.matrix(read.csv('./mouse_gene_names.csv', check.names=FALSE)))
mouse_gene_list = c(mouse_gene_df)
# print(mouse_gene_df)
# 使用homologene函进行转换
# genelist是要转换的基因列表
# inTax是输入的基因列表所属的物种号，10090是小鼠
# outTax是要转换成的物种号，9606是人
common_gene_df = homologene(mouse_gene_df, inTax = 10090, outTax = 9606)
write.csv(common_gene_df, './commom_gene_names.csv')

