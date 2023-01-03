'''
    对数据集内部的实验做自动化脚本
    分层抽样，五五分
    放进data目录
    利用get_sm构图
    不需要做MNN，所以在py文件中直接做preprocess就行(Median norm + gene selction)
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import os

def process_and_save(base_path, project, save_name):
    for i, p in enumerate(project):
        data_path = os.path.join(base_path, p, 'data.csv')
        label_path = os.path.join(base_path, p, 'label.csv')
        data = pd.read_csv(data_path, index_col=0)
        label = pd.read_csv(label_path)
        label.columns = ['type']
        # 去掉未标注的细胞
        idx = (~label['type'].isin(['Unassigned', 'unclear', 'not', 'co-expression', 'unclassified', 'not applicable', 'unclassified cell', 'co-expression cell', 'unclassified endocrine cell'])).tolist()
        print(label.shape)
        label = label.iloc[idx]
        print(label.shape)
        data = data.iloc[idx, :]

        # 对特殊的数据做处理
        if p in ['10X_V3', 'CEL_Seq2', 'DropSeq', 'InDrop', 'Seq_Well', 'Smart_seq2', 'GSE81608', 'GSE85241']:
            # platform部分数据要处理
            data.columns = [x.split("_")[1] for x in data.columns.tolist()]
        if p == 'E-MTAB-5061':
            label['type'] = label['type'].apply(lambda x: x.split()[0])

        print(data.shape, label.shape)
        print(np.unique(label['type']))

        train_x, test_x, train_y, test_y = train_test_split(data,
                                                            label['type'],
                                                            shuffle=True,
                                                            stratify=label['type'],
                                                            test_size=0.5,
                                                            random_state=0
                                                            )
        train_y = pd.DataFrame({
            'type': train_y
        })
        test_y = pd.DataFrame({
            'type': test_y
        })

        if not os.path.exists(save_name[i]):
            os.mkdir(save_name[i])
            os.mkdir(os.path.join(save_name[i], "data"))
            os.mkdir(os.path.join(save_name[i], "data", "ref"))
            os.mkdir(os.path.join(save_name[i], "data", "query"))

        train_x.to_csv(os.path.join(save_name[i], "data/ref/data_1.csv"))
        test_x.to_csv(os.path.join(save_name[i], "data/query/data_1.csv"))
        train_y.to_csv(os.path.join(save_name[i], "data/ref/label_1.csv"), index=False)
        test_y.to_csv(os.path.join(save_name[i], "data/query/label_1.csv"), index=False)

# 先处理species
# base_path = 'E:/YuAnHuang/kevislin/data/species'
# project = ['E-MTAB-5061']
# save_name = ['emtab5061']
# process_and_save(base_path, project, save_name)


# base_path = 'E:/YuAnHuang/kevislin/data/species/GSE84133'
# project = ['raw_mouse_data', 'raw_human_data']
# save_name = ['gse84133_mouse', 'gse84133_human']
# process_and_save(base_path, project, save_name)

# Platform
base_path = 'E:/YuAnHuang/kevislin/data/platform'
project = ['10X_V3', 'CEL_Seq2', 'DropSeq', 'InDrop', 'Seq_Well', 'Smart_seq2', 'GSE81608', 'GSE85241']
save_name = ['10x_v3', 'cel_seq', 'dropseq', 'indrop', 'seq_well', 'smart_seq', 'gse81608', 'gse85241']
process_and_save(base_path, project, save_name)