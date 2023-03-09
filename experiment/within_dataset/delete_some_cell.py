import os
import pandas as pd
project = 'He_Calvarial_Bone'
ref_data = pd.read_csv(os.path.join(project, 'raw_data', 'ref', 'data_1.csv'), index_col=0)
ref_label = pd.read_csv(os.path.join(project, 'raw_data', 'ref', 'label_1.csv'))

query_data = pd.read_csv(os.path.join(project, 'raw_data', 'query', 'data_1.csv'), index_col=0)
query_label = pd.read_csv(os.path.join(project, 'raw_data', 'query', 'label_1.csv'))


# delete_cell = ['sertoli cell', "late primary S'cytes", "elongated S'tids"]
delete_cell = ["Myoblast", "PMSC2", "mig_NC"]

ref_idx = (~ref_label.iloc[:, 0].isin(delete_cell)).tolist()
query_idx = (~query_label.iloc[:, 0].isin(delete_cell)).tolist()

ref_data = ref_data.iloc[ref_idx, :]
ref_label = ref_label.iloc[ref_idx, :]


query_data = query_data.iloc[query_idx, :]
query_label = query_label.iloc[query_idx, :]

# save
ref_data.to_csv(os.path.join(project, 'raw_data', 'ref', 'data_1.csv'))
ref_label.to_csv(os.path.join(project, 'raw_data', 'ref', 'label_1.csv'), index=False)

query_data.to_csv(os.path.join(project, 'raw_data', 'query', 'data_1.csv'))
query_label.to_csv(os.path.join(project, 'raw_data', 'query', 'label_1.csv'), index=False)

