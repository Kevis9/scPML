import pandas as pd
data = pd.read_csv('raw_data/ref/data_1.csv', index_col=0)
label = pd.read_csv('raw_data/ref/label_1.csv')

gsehuman_data = pd.read_csv('E:/YuAnHuang/kevislin/data/species/GSE84133/raw_data_with_same_gene_from_singlecellNet/human/data.csv', index_col=0)
gsehuman_label = pd.read_csv('E:/YuAnHuang/kevislin/data/species/GSE84133/raw_data_with_same_gene_from_singlecellNet/human/label.csv')

emtab_data = pd.read_csv('E:/YuAnHuang/kevislin/data/species/E-MTAB-5061/data.csv', index_col=0)
emtab_label = pd.read_csv('E:/YuAnHuang/kevislin/data/species/E-MTAB-5061/label.csv')

gse85241_data = pd.read_csv('E:/YuAnHuang/kevislin/data/platform/GSE85241/data.csv', index_col=0)
gse85241_label = pd.read_csv('E:/YuAnHuang/kevislin/data/platform/GSE85241/label.csv')


gse81608_data = pd.read_csv('E:/YuAnHuang/kevislin/data/platform/GSE81608/data.csv', index_col=0)
gse81608_label = pd.read_csv('E:/YuAnHuang/kevislin/data/platform/GSE81608/label.csv')

emtab_label.columns = ['type']
gse81608_label.columns = ['type']
gse85241_label.columns = ['type']
gsehuman_label.columns = ['type']
gsehuman_label['type'] = [x.split()[0].lower() for x in gsehuman_label.iloc[:, 0].tolist()]
emtab_label['type'] = [x.split()[0].lower() for x in emtab_label.iloc[:, 0].tolist()]
gse85241_label['type'] = [x.split()[0].lower() for x in gse85241_label.iloc[:, 0].tolist()]
gse81608_label['type'] = [x.split()[0].lower() for x in gse81608_label.iloc[:, 0].tolist()]

# 基因交集
gs = data.columns.tolist()


idx = gsehuman_label.iloc[:, 0].isin(label.iloc[:, 0].tolist()).tolist()
gsehuman_data = gsehuman_data[gs]
gsehuman_data = gsehuman_data.iloc[idx, :]
gsehuman_label = gsehuman_label.iloc[idx, :]


idx = emtab_label.iloc[:, 0].isin(label.iloc[:, 0].tolist()).tolist()
emtab_data = emtab_data[gs]
emtab_data = emtab_data.iloc[idx, :]
emtab_label = emtab_label.iloc[idx, :]

idx = gse85241_label.iloc[:, 0].isin(label.iloc[:, 0].tolist()).tolist()
gse85241_data = gse85241_data[gs]
gse85241_data = gse85241_data.iloc[idx, :]
gse85241_label = gse85241_label.iloc[idx, :]

idx = gse81608_label.iloc[:, 0].isin(label.iloc[:, 0].tolist()).tolist()
gse81608_data = gse81608_data[gs]
gse81608_data = gse81608_data.iloc[idx, :]
gse81608_label = gse81608_label.iloc[idx, :]

print(gsehuman_data.shape)
print(gsehuman_label.shape)

print(emtab_data.shape)
print(emtab_label.shape)

print(gse85241_data.shape)
print(gse85241_label.shape)

# query_data = pd.read_csv('raw_data/query/data_1.csv', index_col=0)
# print(query_data.shape)

combine_label = pd.concat([emtab_label, gse81608_label, gse85241_label, gsehuman_label], axis=0)
combine_data = pd.concat([emtab_data, gse81608_data, gse85241_data, gsehuman_data], axis=0)

print(combine_data.shape)
print(combine_label.shape)

gsehuman_data.to_csv('query_split_data/gsehuman_data.csv')
emtab_data.to_csv('query_split_data/emtab_data.csv')
gse85241_data.to_csv('query_split_data/gse85241_data.csv')
gse81608_data.to_csv('query_split_data/gse81608_data.csv')

gsehuman_label.to_csv('query_split_data/gsehuman_label.csv', index=False)
emtab_label.to_csv('query_split_data/emtab_label.csv', index=False)
gse85241_label.to_csv('query_split_data/gse85241_label.csv', index=False)
gse81608_label.to_csv('query_split_data/gse81608_label.csv', index=False)

combine_data.to_csv("query_split_data/combine_data.csv")
combine_label.to_csv('query_split_data/combine_label.csv', index=False)

