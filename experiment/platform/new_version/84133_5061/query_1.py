import sys
import torch

sys.path.append('../../../..')
import os
os.system("wandb disabled")
from MVCC.util import sc_normalization, construct_graph_with_knn,\
    read_data_label_h5, read_similarity_mat_h5, encode_label, show_result, pre_process
from MVCC.model import MVCCModel
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import wandb

# 数据配置
data_config = {
    'root_path': '.', # data.h5 path
    'ref_name': 'cel_seq2',
    'query_name': 'dropseq',
    'ref_key': 'ref_1',
    'query_key': 'query_1',
    'project': 'platform',
}

parameter_config = {
    'gcn_middle_out': 2048,  # GCN中间层维数
    'lsd': 512,  # CPM_net latent space dimension
    'lamb': 100,  # classfication loss的权重
    'epoch_cpm_ref': 500,
    'epoch_cpm_query': 50,
    'exp_mode': 3, # 1: start from scratch,
                   # 2: multi ref ,
                   # 3: gcn model exists, train cpm model and classifier
    'nf': 3000,
    'classifier_name':"FC",
    # 不太重要参数
    'batch_size_classifier': 256,  # CPM中重构和分类的batch size
    'epoch_gcn': 1000,  # Huang gcn 训练的epoch
    'epoch_classifier': 500,
    'patience_for_classifier': 20,
    'patience_for_gcn': 200,  # 训练GCN的时候加入一个早停机制
    'patience_for_cpm_ref': 50, # cpm train ref 早停patience
    'patience_for_cpm_query': 50, # query h 早停patience
    'k_neighbor': 3,  # GCN 图构造的时候k_neighbor参数
    'mask_rate': 0.3,
    'gamma': 1,
    'test_size': 0.2,
    'show_result': False,
}


def main_process():
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
    ref_norm_data, query_norm_data = pre_process(ref_data, query_data, ref_label, nf=parameter_config['nf'])

    # np.savetxt("ref_data.csv", ref_norm_data, delimiter=',')
    # np.savetxt("query_data.csv", query_norm_data, delimiter=',')



    # exit()
    # ref_norm_data = sc_normalization(ref_data)
    # query_norm_data = sc_normalization(query_data)

    ref_sm_arr = [read_similarity_mat_h5(data_config['root_path'], data_config['ref_key'] + "/sm_" + str(i + 1)) for i
                  in
                  range(4)]
    query_sm_arr = [read_similarity_mat_h5(data_config['root_path'], data_config['query_key'] + "/sm_" + str(i + 1)) for
                    i in
                    range(4)]

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
                  classifier_name=parameter_config['classifier_name']
                  )
    pred = mvccmodel.predict(query_norm_data, query_sm_arr, parameter_config['epoch_cpm_query'],
                             parameter_config['k_neighbor'])
    pred_cpm = mvccmodel.predict_with_cpm()

    # 因为打乱了train数据集，所以这里记录下raw label
    ref_raw_label = enc.inverse_transform(ref_label)
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
    print("cpm acc is {:.3f}".format(cpm_acc))
    ret = {
        'ref_out': ref_out,
        'query_out': query_out,
        'ref_raw_data': ref_data,
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