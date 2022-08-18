import pandas as pd



ref_data = pd.read_csv(r'F:\yuanhuang\kevislin\Cell_Classification\experiment2\platform\task1\data\ref\data_1.csv', index_col=0)
query_data = pd.read_csv(r'F:\yuanhuang\kevislin\Cell_Classification\experiment2\platform\task1\data\query\data_1.csv', index_col=0)
ref_data.columns = [x.split('_')[1] for x in ref_data.columns.tolist()]
query_data.columns = [x.split('_')[1] for x in query_data.columns.tolist()]

ref_data.to_csv(r'F:\yuanhuang\kevislin\Cell_Classification\experiment2\platform\task1\data\ref\data_1.csv')
query_data.to_csv(r'F:\yuanhuang\kevislin\Cell_Classification\experiment2\platform\task1\data\query\data_1.csv')

