import sys
import random
import torch

sys.path.append('../../../..')
import os
from MVCC.util import mean_norm, construct_graph_with_knn, \
    read_data_label_h5, read_similarity_mat_h5, encode_label, show_result, pre_process, z_score_scale, \
    check_out_similarity_matrix, construct_graph, setup_seed
from MVCC.model import MVCCModel
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 数据配置
data_config = {
    'root_path': '.',  # data.h5 path
    'ref_key': 'ref_1',
    'query_key': 'query_1',
}

parameter_config = {
    'gcn_middle_out': 1024,  # layer size of GCN
    'lsd': 512,  # CPM_net latent space dimension
    'lamb': 3000,  # classification loss weight
    'epoch_cpm_ref': 300,
    'epoch_cpm_query': 30,
    'exp_mode': 1,  # 1: training GCN,cpm and classifier
                    # 2: multi ref ,
                    # 3: gcn model exists, train cpm model and classifier
    'classifier_name': "FC",
    'batch_size_classifier': 256,
    'epoch_gcn': 500,
    'epoch_classifier': 100,
    'k_neighbor': 30,
    'mask_rate': 0.3,
    'gamma': 1,
    'show_result': True,
    'views': [1, 2, 3]
}

import pickle

with open("hyper_parameters", 'wb') as f:
    pickle.dump(parameter_config, f)


def main_process():
    # random seed setting
    setup_seed(20)

    ref_data, ref_label = read_data_label_h5(data_config['root_path'], data_config['ref_key'])
    query_data, query_label = read_data_label_h5(data_config['root_path'], data_config['query_key'])
    ref_data = ref_data.astype(np.float64)
    query_data = query_data.astype(np.float64)

    ref_norm_data = ref_data
    query_norm_data = query_data

    # similarity matrices
    ref_sm_arr = [read_similarity_mat_h5(data_config['root_path'], data_config['ref_key'] + "/sm_" + str(i + 1)) for i
                  in
                  parameter_config['views']]
    query_sm_arr = [read_similarity_mat_h5(data_config['root_path'], data_config['query_key'] + "/sm_" + str(i + 1)) for
                    i in
                    parameter_config['views']]

    if parameter_config['exp_mode'] == 2:
        # multi ref
        mvccmodel = torch.load('model/mvccmodel_' + data_config['query_key'] + ".pt")
        ref_label, query_label = mvccmodel.label_encoder.transform(ref_label), mvccmodel.label_encoder.transform(
            query_label)
        enc = mvccmodel.label_encoder
    else:
        ref_label, query_label, enc = encode_label(ref_label, query_label)
        mvccmodel = MVCCModel(
            lsd=parameter_config['lsd'],
            class_num=len(set(ref_label)),
            view_num=len(ref_sm_arr),
            save_path=data_config['root_path'],
            label_encoder=enc,

        )

    mvccmodel.fit(ref_norm_data, ref_sm_arr, ref_label,
                  gcn_input_dim=ref_norm_data.shape[1],
                  gcn_middle_out=parameter_config['gcn_middle_out'],
                  lamb=parameter_config['lamb'],
                  epoch_gcn=parameter_config['epoch_gcn'],
                  epoch_cpm_ref=parameter_config['epoch_cpm_ref'],
                  epoch_classifier=parameter_config['epoch_classifier'],
                  batch_size_classifier=parameter_config['batch_size_classifier'],
                  mask_rate=parameter_config['mask_rate'],
                  exp_mode=parameter_config['exp_mode'],
                  k_neighbor=parameter_config['k_neighbor'],
                  classifier_name=parameter_config['classifier_name']
                  )

    pred = mvccmodel.predict(query_norm_data, query_sm_arr, parameter_config['epoch_cpm_query'],
                             k_neighbor=parameter_config['k_neighbor'])

    ref_raw_label = enc.inverse_transform(ref_label)

    # GCN embeddings
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

    ret = {
        'ref_gcn_embeddings': ref_gcn_embeddings,
        'query_gcn_embeddings': query_gcn_embeddings,
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
        show_result(ret, "result")
    return ret


ret = main_process()
acc = accuracy_score(ret['pred'], ret['query_label'])
print("pred acc is {:.3f}".format(acc))
