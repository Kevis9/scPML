import pandas as pd
import numpy as np
import os

def split_data(data, label):
    ref_1_idx = []
    ref_2_idx = []
    cell_dic = dict(label.iloc[:, 0].value_counts().map(lambda x: int(x * (0.5))))
    label_np = label.to_numpy().reshape(-1)
    # print(label_np)
    for key in cell_dic.keys():
        idx = np.array(np.where(label_np == key)).squeeze().tolist()
        # print(len(idx), key)
        np.random.shuffle(idx)
        ref_1_idx += idx[:cell_dic[key]]
        ref_2_idx += idx[cell_dic[key]:]

    ref_1_data = data.iloc[ref_1_idx, :]
    ref_2_data = data.iloc[ref_2_idx, :]
    ref_1_label = label.iloc[ref_1_idx]
    ref_2_label = label.iloc[ref_2_idx]

    return (ref_1_data, ref_1_label, ref_2_data, ref_2_label)

src_path = r"E:\YuAnHuang\kevislin\data\within_dataset"

projects = sorted(os.listdir(src_path))

projects = [
    'GSE98638',
    'GSE99254'
]
# 标签有问题的表达
except_type = ["No Class", "Low Quality", "nan"]

for proj in projects:
    data_path = os.path.join(src_path, proj, "data.csv")
    label_path = os.path.join(src_path, proj, "label.csv")
    label = pd.read_csv(label_path)
    data = pd.read_csv(data_path, index_col=0)

    # 去掉标签有问题的细胞
    idx = (~label.iloc[:, 0].isin(except_type)).tolist()
    data = data.iloc[idx, :]
    label = label.iloc[idx, :]

    # 做一个质控, counts<=500的细胞直接去掉
    # data_np = data.to_numpy()
    # data_row_sum = data_np.sum(axis=1).squeeze()
    # idx = np.where(data_row_sum > 500)
    # print()


    # split cells randomly
    ref_data, ref_label, query_data, query_label = split_data(data, label)

    print("details about {:}".format(proj))
    print(ref_data.shape)
    print(ref_label.shape)
    print(ref_label.iloc[:, 0].value_counts())
    print(query_data.shape)
    print(query_label.shape)
    print(query_label.iloc[:, 0].value_counts())

    # save
    if not os.path.exists(proj):
        os.makedirs(proj)
        os.makedirs(os.path.join(proj, "raw_data"))
        os.makedirs(os.path.join(proj, "raw_data", "ref"))
        os.makedirs(os.path.join(proj, "raw_data", "query"))

    ref_data.to_csv(os.path.join(proj, "raw_data", "ref", "data_1.csv"))
    ref_label.to_csv(os.path.join(proj, "raw_data", "ref", "label_1.csv"), index=False)
    query_data.to_csv(os.path.join(proj, "raw_data", "query", "data_1.csv"))
    query_label.to_csv(os.path.join(proj, "raw_data", "query", "label_1.csv"), index=False)
