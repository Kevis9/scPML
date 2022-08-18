import os
from calendar import c
from operator import index
import pandas as pd

ref_save_path = 'experiment/platform/task_10x_all/data/ref'
query_save_path = 'experiment/platform/task_10x_all/data/query'

ref_path = r'F:\yuanhuang\kevislin\data\platform\10X_V3'

# query1_path = r'F:\yuanhuang\kevislin\data\platform\CEL_Seq2'
query2_path = r'F:\yuanhuang\kevislin\data\platform\InDrop'
query3_path = r'F:\yuanhuang\kevislin\data\platform\Seq_Well'
# query4_path = r'F:\yuanhuang\kevislin\data\platform\Smart_seq2'
query5_path = r'F:\yuanhuang\kevislin\data\platform\DropSeq'


ref_label = pd.read_csv(os.path.join(ref_path, 'label.csv'))
# query1_label = pd.read_csv(os.path.join(query1_path, 'label.csv'))
query2_label = pd.read_csv(os.path.join(query2_path, 'label.csv'))
query3_label = pd.read_csv(os.path.join(query3_path, 'label.csv'))
# query4_label = pd.read_csv(os.path.join(query4_path, 'label.csv'))
query5_label = pd.read_csv(os.path.join(query5_path, 'label.csv'))

label_key = 'CellType'

ref_idx = (~ref_label[label_key].isin(['Unassigned'])).tolist()
ref_label = ref_label.iloc[ref_idx, :]
ref_label_list = ref_label[label_key].tolist()

# query1_idx = query1_label[label_key].isin(ref_label_list).tolist()
query2_idx = query2_label[label_key].isin(ref_label_list).tolist()
query3_idx = query3_label[label_key].isin(ref_label_list).tolist()
# query4_idx = query4_label[label_key].isin(ref_label_list).tolist()
query5_idx = query5_label[label_key].isin(ref_label_list).tolist()

# query1_label = query1_label.iloc[query1_idx, :]
query2_label = query2_label.iloc[query2_idx, :]
query3_label = query3_label.iloc[query3_idx, :]
# query4_label = query4_label.iloc[query4_idx, :]
query5_label = query5_label.iloc[query5_idx, :]

print("ref label set :", sorted(list(set(ref_label[label_key].tolist()))))
# print("query 1 label set :", sorted(list(set(query1_label[label_key].tolist()))))
print("query 2 label set :", sorted(list(set(query2_label[label_key].tolist()))))
print("query 3 label set :", sorted(list(set(query3_label[label_key].tolist()))))
# print("query 4 label set :", sorted(list(set(query4_label[label_key].tolist()))))
print("query 5 label set :", sorted(list(set(query5_label[label_key].tolist()))))



ref_data = pd.read_csv(os.path.join(ref_path, 'data.csv'), index_col=0)
# query1_data = pd.read_csv(os.path.join(query1_path, 'data.csv'), index_col=0)
query2_data = pd.read_csv(os.path.join(query2_path, 'data.csv'), index_col=0)
query3_data = pd.read_csv(os.path.join(query3_path, 'data.csv'), index_col=0)
# query4_data = pd.read_csv(os.path.join(query4_path, 'data.csv'), index_col=0)
query5_data = pd.read_csv(os.path.join(query5_path, 'data.csv'), index_col=0)

ref_data.columns = [x.split("_")[1] for x in ref_data.columns.tolist()]
# query1_data.columns = [x.split("_")[1] for x in query1_data.columns.tolist()]
query2_data.columns = [x.split("_")[1] for x in query2_data.columns.tolist()]
query3_data.columns = [x.split("_")[1] for x in query3_data.columns.tolist()]
# query4_data.columns = [x.split("_")[1] for x in query4_data.columns.tolist()]
query5_data.columns = [x.split("_")[1] for x in query5_data.columns.tolist()]

ref_data = ref_data.iloc[ref_idx, :]
# query1_data = query1_data.iloc[query1_idx, :]
query2_data = query2_data.iloc[query2_idx, :]
query3_data = query3_data.iloc[query3_idx, :]
# query4_data = query4_data.iloc[query4_idx, :]
query5_data = query5_data.iloc[query5_idx, :]
#
print("ref data shape ", ref_data.shape)
# print("query 1 data shape", query1_data.shape)
print("query 2 data shape", query2_data.shape)
print("query 3 data shape", query3_data.shape)
# print("query 4 data shape", query4_data.shape)
print("query 5 data shape", query5_data.shape)

ref_data.to_csv(os.path.join(ref_save_path, 'data_1.csv'))
# query1_data.to_csv(os.path.join(query_save_path, 'data_1.csv'))
query2_data.to_csv(os.path.join(query_save_path, 'data_2.csv'))
query3_data.to_csv(os.path.join(query_save_path, 'data_3.csv'))
# query4_data.to_csv(os.path.join(query_save_path, 'data_4.csv'))
query5_data.to_csv(os.path.join(query_save_path, 'data_5.csv'))
#
ref_label.to_csv(os.path.join(ref_save_path, 'label_1.csv'), index=False)
# query1_label.to_csv(os.path.join(query_save_path, 'label_1.csv'), index=False)
query2_label.to_csv(os.path.join(query_save_path, 'label_2.csv'), index=False)
query3_label.to_csv(os.path.join(query_save_path, 'label_3.csv'), index=False)
# query4_label.to_csv(os.path.join(query_save_path, 'label_4.csv'), index=False)
query5_label.to_csv(os.path.join(query_save_path, 'label_5.csv'), index=False)