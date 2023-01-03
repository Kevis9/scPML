import pandas as pd
query_1 = pd.read_csv("emtab_data.csv", index_col=0)
query_2 = pd.read_csv("gse81608_data.csv", index_col=0)
query_3 = pd.read_csv("gse85241_data.csv", index_col=0)
query_4 = pd.read_csv("gsehuman_data.csv", index_col=0)

combine_data = pd.concat([query_1, query_2, query_3, query_4], axis=0)
combine_data.to_csv('combine_data.csv')
