import pandas as pd

# raw data
data1 = pd.read_csv('CEL_Seq2\\CEL_Seq2_data.csv', index_col=0)
label1 = pd.read_csv('CEL_Seq2\\CEL_Seq2_label.csv')
gene = data1.columns.map(lambda x: x.split("_")[1]).tolist()
data1.columns = gene


data2 = pd.read_csv('10X_V3\\10x_v3_data.csv', index_col=0)
label2 = pd.read_csv('10X_V3\\10x_v3_label.csv')
data2.columns = gene

data3 = pd.read_csv('InDrop\\InDrop_data.csv', index_col=0)
label3 = pd.read_csv('InDrop\\InDrop_label.csv')
data3.columns = gene

data4 = pd.read_csv('DropSeq\\DropSeq_data.csv', index_col=0)
label4 = pd.read_csv('DropSeq\\DropSeq_label.csv')
data4.columns = gene

data5 = pd.read_csv('Seq_Well\\Seq_Well_data.csv', index_col=0)
label5 = pd.read_csv('Seq_Well\\Seq_Well_label.csv')
data5.columns = gene

data6 = pd.read_csv('Smart_seq2\\smart_seq2_data.csv', index_col=0)
label6 = pd.read_csv('Smart_seq2\\smart_seq2_label.csv')
data6.columns = gene








def save_data_label(data, label, c_type, path1, path2):
    idx = label['CellType'].isin(c_type).tolist()
    new_data = data.iloc[idx, :]
    new_label = label.iloc[idx, :]
    new_data.to_csv(path1)
    new_label.to_csv(path2, index=False)
    # new_data.to_csv(path+"\\data.csv")
    # new_label.to_csv(path+"\\label.csv", index=False)


common_type = set(label1['CellType'].tolist()) & set(label2['CellType'].tolist()) & \
              set(label3['CellType'].tolist()) & set(label4['CellType'].tolist()) & \
              set(label5['CellType'].tolist()) & set(label6['CellType'].tolist())

save_data_label(data1, label1, common_type,"data1.csv", "label1.csv")
save_data_label(data2, label2, common_type,"data2.csv", "label2.csv")
save_data_label(data3, label3, common_type,"data3.csv", "label3.csv")
save_data_label(data4, label4, common_type,"data4.csv", "label4.csv")
save_data_label(data5, label5, common_type,"data5.csv", "label5.csv")
save_data_label(data6, label6, common_type,"data6.csv", "label6.csv")

# task1: ref: data1
# common_type = list(set(label1['CellType'].tolist()))
# save_data_label(data2, label2, common_type, "task1\\query\\query_1")
# save_data_label(data3, label3, common_type, "task1\\query\\query_2")
# save_data_label(data4, label4, common_type, "task1\\query\\query_3")
# save_data_label(data5, label5, common_type, "task1\\query\\query_4")
# save_data_label(data6, label6, common_type, "task1\\query\\query_5")
# data1.to_csv("task1\\ref\\data.csv")
# label1.to_csv("task1\\ref\\label.csv")
#
#
# # task2: ref: data2
# common_type = list(set(label2['CellType'].tolist()))
# save_data_label(data1, label1, common_type, "task2\\query\\query_1")
# save_data_label(data3, label3, common_type, "task2\\query\\query_2")
# save_data_label(data4, label4, common_type, "task2\\query\\query_3")
# save_data_label(data5, label5, common_type, "task2\\query\\query_4")
# save_data_label(data6, label6, common_type, "task2\\query\\query_5")
# data2.to_csv("task2\\ref\\data.csv")
# label2.to_csv("task2\\ref\\label.csv")
#
# # task3 : ref: data3
# common_type = list(set(label3['CellType'].tolist()))
# save_data_label(data1, label1, common_type, "task3\\query\\query_1")
# save_data_label(data2, label2, common_type, "task3\\query\\query_2")
# save_data_label(data4, label4, common_type, "task3\\query\\query_3")
# save_data_label(data5, label5, common_type, "task3\\query\\query_4")
# save_data_label(data6, label6, common_type, "task3\\query\\query_5")
# data3.to_csv("task3\\ref\\data.csv")
# label3.to_csv("task3\\ref\\label.csv")
#
#
# # task4 : ref: data4
# common_type = list(set(label4['CellType'].tolist()))
# save_data_label(data1, label1, common_type, "task4\\query\\query_1")
# save_data_label(data2, label2, common_type, "task4\\query\\query_2")
# save_data_label(data3, label3, common_type, "task4\\query\\query_3")
# save_data_label(data5, label5, common_type, "task4\\query\\query_4")
# save_data_label(data6, label6, common_type, "task4\\query\\query_5")
# data4.to_csv("task4\\ref\\data.csv")
# label4.to_csv("task4\\ref\\label.csv")
#
# # task5 : ref: data5
# common_type = list(set(label5['CellType'].tolist()))
# save_data_label(data1, label1, common_type, "task5\\query\\query_1")
# save_data_label(data2, label2, common_type, "task5\\query\\query_2")
# save_data_label(data3, label3, common_type, "task5\\query\\query_3")
# save_data_label(data4, label4, common_type, "task5\\query\\query_4")
# save_data_label(data6, label6, common_type, "task5\\query\\query_5")
# data5.to_csv("task5\\ref\\data.csv")
# label5.to_csv("task5\\ref\\label.csv")
#
# # task6 : ref: data6
# common_type = list(set(label6['CellType'].tolist()))
# save_data_label(data1, label1, common_type, "task6\\query\\query_1")
# save_data_label(data2, label2, common_type, "task6\\query\\query_2")
# save_data_label(data3, label3, common_type, "task6\\query\\query_3")
# save_data_label(data4, label4, common_type, "task6\\query\\query_4")
# save_data_label(data5, label5, common_type, "task6\\query\\query_5")
# data6.to_csv("task6\\ref\\data.csv")
# label6.to_csv("task6\\ref\\label.csv")












