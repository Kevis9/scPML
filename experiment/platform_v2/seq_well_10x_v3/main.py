import sys
import random

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

parameter_config = {
    'gcn_middle_out': 1024,  # GCN中间层维数
    'lsd': 512,  # CPM_net latent space dimension
    'lamb': 3000,  # classfication loss的权重
    'epoch_cpm_ref': 300,
    'epoch_cpm_query': 30,
    'exp_mode': 1, # 1: start from scratch,
                   # 2: multi ref ,
                   # 3: gcn model exists, train cpm model and classifier
    'classifier_name':"FC",
    # 不太重要参数
    'batch_size_classifier': 256,  # CPM中重构和分类的batch size
    'epoch_gcn': 500,  # Huang gcn 训练的epoch
    'epoch_classifier': 100,
    'patience_for_classifier': 50,
    'patience_for_gcn': 200,  # 训练GCN的时候加入一个早停机制
    'patience_for_cpm_ref': 300, # cpm train ref 早停patience
    'patience_for_cpm_query': 200, # query h 早停patience
    'k_neighbor': 30,  # GCN 图构造的时候k_neighbor参数
    'mask_rate': 0.3,
    'gamma': 1,
    'test_size': 0.2,
    'show_result': True,
    'views':[1,2,3]
}

import pickle
with open("hyper_parameters", 'wb') as f:
    pickle.dump(parameter_config, f)

def main_process():
    # 设置torch的seed
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
                  parameter_config['views']]
    query_sm_arr = [read_similarity_mat_h5(data_config['root_path'], data_config['query_key'] + "/sm_" + str(i + 1)) for
                    i in
                    parameter_config['views']]

    for i in range(len(ref_sm_arr)):
        check_out_similarity_matrix(ref_sm_arr[i], ref_label, k=5, sm_name='ref_'+str(i+1))

    for i in range(len(query_sm_arr)):
        check_out_similarity_matrix(query_sm_arr[i], query_label, k=5, sm_name='query_' + str(i + 1))
    exit()

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
            save_path=data_config['root_path'],
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
                  gamma=parameter_config['gamma'],
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
    np.save("result/cpm_preds", pred_cpm)
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
        show_result(ret, "result")
    run.finish()
    return ret


ret = main_process()
acc = accuracy_score(ret['pred'], ret['query_label'])
print("pred acc is {:.3f}".format(acc))