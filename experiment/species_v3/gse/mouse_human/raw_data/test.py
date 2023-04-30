import pandas as pd
from sklearn.feature_selection import SelectKBest
data = pd.read_csv('../data/ref/data_1.csv', index_col=0)
raw_data = pd.read_csv('ref/data_1.csv', index_col=0)
label = pd.read_csv('ref/label_1.csv')
# 检查一下selectKBest和R的preprocess带来的效果是否一样
selector = SelectKBest(k=2000)
selector.fit(raw_data, label)
egen = set(raw_data.columns[selector.get_support()].tolist())
print(set(data.columns.tolist()) == egen)