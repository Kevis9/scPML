import sys

import pandas as pd
import torch

sys.path.append('../../../..')
import os
os.system("wandb disabled")
from MVCC.util import mean_norm, construct_graph_with_knn, \
    read_data_label_h5, read_similarity_mat_h5, encode_label, show_result, pre_process, z_score_scale, \
    check_out_similarity_matrix, construct_graph, setup_seed
from MVCC.model import MVCCModel
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import wandb

# 数据配置
data_config = {
    'root_path': '.', # data.h5 path
    'ref_name': 'cel_seq2',
    'query_name': 'smart_seq',
    'ref_key': 'ref_1',
    'query_key': 'query_1',
    'project': 'platform',
}

ori_parameter_config = {
    'gcn_middle_out': 1024,  # GCN中间层维数
    'lsd': 512,  # CPM_net latent space dimension
    'lamb': 5000,  # classfication loss的权重
    'epoch_cpm_ref': 500,
    'epoch_cpm_query': 30,
    'exp_mode': 1, # 1: start from scratch,
                   # 2: multi ref ,
                   # 3: gcn model exists, train cpm model and classifier
    'classifier_name':"FC",
    # 不太重要参数
    'batch_size_classifier': 256,  # CPM中重构和分类的batch size
    'epoch_gcn': 500,  # Huang gcn 训练的epoch
    'epoch_classifier': 500,
    'patience_for_classifier': 50,
    'patience_for_gcn': 200,  # 训练GCN的时候加入一个早停机制
    'patience_for_cpm_ref': 300, # cpm train ref 早停patience
    'patience_for_cpm_query': 200, # query h 早停patience
    'k_neighbor': 3,  # GCN 图构造的时候k_neighbor参数
    'mask_rate': 0.3,
    'gamma': 1,
    'test_size': 0.2,
    'show_result': False,
    'view_ranges': [0, 1, 2, 3],
    'result_path': '.'
}



def main_process(parameter_config):
    setup_seed(20)
    run = wandb.init(project="cell_classify_" + data_config['project'],
                     entity="kevislin",
                     config={"config": parameter_config, "data_config": data_config},
                     tags=[data_config['ref_name'] + '-' + data_config['query_name'], data_config['project']],
                     reinit=True)

    # 数据准备
    ref_data, ref_label = read_data_label_h5(data_config['root_path'], data_config['ref_key'])
    query_data, query_label = read_data_label_h5(data_config['root_path'], data_config['query_key'])
    ref_data = ref_data.astype(np.float64)
    query_data = query_data.astype(np.float64)
    # ref_norm_data, query_norm_data = pre_process(ref_data, query_data, ref_label, nf=2000)
    # ref_norm_data = sc_normalization(ref_data)
    # query_norm_data = sc_normalization(query_data)
    ref_norm_data = ref_data
    query_norm_data = query_data
    ref_sm_arr = [read_similarity_mat_h5(data_config['root_path'], data_config['ref_key'] + "/sm_" + str(i + 1)) for i
                  in
                  parameter_config['view_ranges']]
    query_sm_arr = [read_similarity_mat_h5(data_config['root_path'], data_config['query_key'] + "/sm_" + str(i + 1)) for
                    i in
                    parameter_config['view_ranges']]

    # for i in range(len(ref_sm_arr)):
    #     check_out_similarity_matrix(ref_sm_arr[i], ref_label, k=parameter_config['k_neighbor'], sm_name='ref_'+str(i+1))
    #
    # for i in range(len(query_sm_arr)):
    #     check_out_similarity_matrix(query_sm_arr[i], query_label, k=parameter_config['k_neighbor'], sm_name='query_' + str(i + 1))
    # exit()

    if parameter_config['exp_mode'] == 2:
        # multi ref
        mvccmodel = torch.load('model/mvccmodel_'+data_config['query_key']+".pt")
        ref_label, query_label = mvccmodel.label_encoder.transform(ref_label), mvccmodel.label_encoder.transform(
            query_label)
        enc = mvccmodel.label_encoder
    else:
        ref_label, query_label, enc = encode_label(ref_label, query_label)

        mvccmodel = MVCCModel(
            lsd=parameter_config['lsd'],
            class_num=len(set(ref_label)),
            view_num=len(ref_sm_arr),
            save_path=parameter_config['result_path'],
            label_encoder=enc,

        )
    mvccmodel.fit(ref_norm_data, ref_sm_arr, ref_label,
                  gcn_input_dim=ref_norm_data.shape[1], gcn_middle_out=parameter_config['gcn_middle_out'],
                  lamb=parameter_config['lamb'], epoch_gcn=parameter_config['epoch_gcn'],
                  epoch_cpm_ref=parameter_config['epoch_cpm_ref'],
                  epoch_classifier=parameter_config['epoch_classifier'],
                  patience_for_classifier=parameter_config['patience_for_classifier'],
                  batch_size_classifier=parameter_config['batch_size_classifier'],
                  mask_rate=parameter_config['mask_rate'],
                  # gamma=parameter_config['gamma'],
                  test_size=parameter_config['test_size'],
                  patience_for_cpm_ref=parameter_config['patience_for_cpm_ref'],
                  patience_for_gcn=parameter_config['patience_for_gcn'],
                  exp_mode=parameter_config['exp_mode'],
                  k_neighbor=parameter_config['k_neighbor'],
                  classifier_name=parameter_config['classifier_name']
                  )
    pred = mvccmodel.predict(query_norm_data, query_sm_arr, parameter_config['epoch_cpm_query'],
                             k_neighbor=parameter_config['k_neighbor'])
    pred_cpm = mvccmodel.predict_with_cpm()

    # 因为打乱了train数据集，所以这里记录下raw label
    ref_raw_label = enc.inverse_transform(ref_label)

    # 展示一下GCN的embeddings
    ref_graph_data = construct_graph(ref_norm_data, ref_sm_arr[0], k=parameter_config['k_neighbor'])
    ref_gcn_embeddings = z_score_scale(mvccmodel.gcn_models[0].get_embedding(ref_graph_data).detach().cpu().numpy())
    query_graph_data = construct_graph(query_norm_data, query_sm_arr[0], k=parameter_config['k_neighbor'])
    query_gcn_embeddings = z_score_scale(mvccmodel.gcn_models[0].get_embedding(query_graph_data).detach().cpu().numpy())

    ref_out, ref_label = mvccmodel.get_ref_embeddings_and_labels()
    # ref_out = mvvcmodel.get_embeddings_with_data(ref_data, ref_sm_arr, 252)
    query_out = mvccmodel.get_query_embeddings()
    ref_out = ref_out.detach().cpu().numpy()
    ref_label = enc.inverse_transform(ref_label.detach().cpu().numpy())
    query_out = query_out.detach().cpu().numpy()
    query_label = enc.inverse_transform(query_label)
    pred = enc.inverse_transform(pred)
    pred_cpm = enc.inverse_transform(pred_cpm)

    cpm_acc = (pred_cpm==query_label).sum() / pred_cpm.shape[0]
    # 保存cpm preds的结果,这里不需要了
    # np.save("result/cpm_preds", pred_cpm)
    print("cpm acc is {:.3f}".format(cpm_acc))
    ret = {
        'ref_gcn_embeddings':ref_gcn_embeddings,
        'query_gcn_embeddings':query_gcn_embeddings,
        'ref_out': ref_out,
        'query_out': query_out,
        'ref_raw_data': ref_data,
        'ref_norm_data': ref_norm_data,
        'query_norm_data': query_norm_data,
        'ref_raw_label': ref_raw_label,
        'ref_label': ref_label,
        'query_raw_data': query_data,
        'query_label': query_label,
        'pred': pred,
        'mvcc_model': mvccmodel
    }
    if parameter_config['show_result']:
        show_result(ret, parameter_config['result_path'])
    run.finish()
    return ret

projs = [
    'gse/mouse_human',
    'gse/human_mouse',
    'mouse_combine',
    'combine_mouse',

    'cel_seq_smart_seq',
    'cel_seq_10x_v3',
    'seq_well_smart_seq',
    'seq_well_drop_seq',
    'seq_well_10x_v3',
    'smart_seq_10x_v3',
    'indrop_drop_seq',
    'indrop_10x_v3',
    'indrop_smart_seq',
    'drop_seq_smart_seq',
    'drop_seq_10x_v3'
]

paths = [
    r'E:\YuAnHuang\kevislin\Cell_Classification\experiment\species_v3',
    r'E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2'
]

hyper_parameters = {
    'k_neighbor': [3, 5, 10, 15, 20, 25, 30],
    'gcn_middle_out': [256, 512, 1024, 2048],
    'lsd': [128, 256, 512, 1024],
    'lamb':  [0, 1, 10, 100, 1000, 3000, 5000],
    'epoch_cpm_ref': [100, 200, 300, 400, 500],
    'epoch_cpm_query': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'epoch_gcn':  [200, 300, 400, 500, 600, 700, 800]
}

for key in hyper_parameters.keys():
    acc_data = pd.DataFrame(columns=[str(para) for para in hyper_parameters[key]], index=projs)
    # 先测试species
    base_path = paths[0]
    for j, proj in enumerate(projs):
        if j >= 4:
            base_path = paths[1]
        data_config['root_path'] = os.path.join(base_path, proj)
        parameter_config = ori_parameter_config.copy()
        parameter_config['show_result'] = False
        for i in range(len(hyper_parameters[key])):
            parameter_config[key] = hyper_parameters[key][i]
            # parameter_config['view_ranges'] = hyper_parameters[key][i]
            if '/' in proj:
                proj_new = proj.split('/')[-1]
            else:
                proj_new = proj


            # if not os.path.exists(proj_new):
            #     os.makedirs(proj_new, exist_ok=True)
            # parameter_config['result_path'] = os.path.join(proj_new, 'result')
            # if not os.path.exists(parameter_config['result_path']):
            #     os.makedirs(parameter_config['result_path'], exist_ok=True)
            # parameter_config['result_path'] = os.path.join(parameter_config['result_path'], str(hyper_parameters[key][i]))
            # if not os.path.exists(parameter_config['result_path']):
            #     os.makedirs(parameter_config['result_path'], exist_ok=True)

            ret = main_process(parameter_config)
            acc = accuracy_score(ret['pred'], ret['query_label'])
            acc_data.loc[proj][str(hyper_parameters[key][i])] = acc
    acc_data.to_csv('acc_data'+'_' + key +'csv')





