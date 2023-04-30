import os.path

import pandas as pd

raw_projs = ['..\\species_v3\\gse\\mouse_human',
             '..\\species_v3\\gse\\human_mouse',
             '..\\species_v3\\mouse_combine',
             '..\\species_v3\\combine_mouse']
projs = ['gsemouse_gsehuman',
         'gsehuman_gsemouse',
         'mouse_combine',
         'combine_mouse']

for i, proj in enumerate(projs):
    ref_data = pd.read_csv(os.path.join(raw_projs[i], 'raw_data', 'ref', 'data_1.csv'), index_col=0)
    query_data = pd.read_csv(os.path.join(raw_projs[i], 'raw_data', 'query', 'data_1.csv'), index_col=0)
    ref_label = pd.read_csv(os.path.join(raw_projs[i], 'raw_data', 'ref', 'label_1.csv'))
    query_label = pd.read_csv(os.path.join(raw_projs[i], 'raw_data', 'query', 'label_1.csv'))

    # 去掉t和schwnn
    ref_idx = (~ref_label.iloc[:, 0].isin(['t', 'schwann', 'T cell'])).tolist()
    query_idx = (~query_label.iloc[:, 0].isin(['t', 'schwann', 'T cell'])).tolist()

    ref_data = ref_data.iloc[ref_idx, :]
    query_data = query_data.iloc[query_idx, :]
    ref_label = ref_label.iloc[ref_idx, :]
    query_label = query_label.iloc[query_idx, :]

    print(ref_label.iloc[:, 0].value_counts())
    print(query_label.iloc[:, 0].value_counts())
    # save
    ref_data.to_csv(os.path.join(proj, 'raw_data', 'ref', 'data_1.csv'))
    query_data.to_csv(os.path.join(proj, 'raw_data', 'query', 'data_1.csv'))
    ref_label.to_csv(os.path.join(proj, 'raw_data', 'ref', 'label_1.csv'), index=False)
    query_label.to_csv(os.path.join(proj, 'raw_data', 'query', 'label_1.csv'), index=False)


print("Finish deleteing")