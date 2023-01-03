import sys
import torch

sys.path.append('../../..')
import os
# os.environ["CUDA_VISIBLE_DEVICES"]='1'
import os.path
from MVCC.util import mean_norm, \
    read_data_label_h5, read_similarity_mat_h5, encode_label, show_result, pre_process
from MVCC.model import MVCCModel
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import wandb

# 数据配置
data_config = {
    # 'root_path': 'F:\\yuanhuang\\kevislin\\data\\species\\task1',
    'root_path': '.',
    'ref_name': 'dropseq',
    # 'query_name': 'E_MTAB_5061: human',
    'query_name': 'all',
    'ref_key': 'ref_1',
    'query_key': 'query_1',
    'project': 'platform',
}
# ['gamma', 'alpha', 'endothelial', 'macrophage', 'ductal', 'delta', 'beta', 'quiescent_stellate']

parameter_config = {
    'gcn_middle_out': 1024,  # GCN中间层维数
    'lsd': 512,  # CPM_net latent space dimension
    'lamb': 3000,  # classfication loss的权重
    'epoch_cpm_ref': 3000,
    'epoch_cpm_query': 600,
    'exp_mode': 1, # 1: start from scratch,
                   # 2: multi ref ,
                   # 3: gcn model exists, train cpm model and classifier
    'classifier_name': 'FC',
    # 不太重要参数
    'batch_size_classifier': 128,  # CPM中重构和分类的batch size
    'epoch_gcn': 1000,  # Huang gcn 训练的epoch
    'epoch_classifier': 2000,
    'patience_for_classifier': 20,
    'patience_for_gcn': 50,  # 训练GCN的时候加入一个早停机制
    'patience_for_cpm_ref': 50, # cpm train ref 早停patience
    'patience_for_cpm_query': 50, # query h 早停patience
    'k_neighbor': 3,  # GCN 图构造的时候k_neighbor参数
    'mask_rate': 0.3,
    'gamma': 1,
    'test_size': 0.2,
}
acc_arr = []
max_acc = 0
cycle = 1


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

    ref_norm_data, query_norm_data = pre_process(ref_data, query_data, ref_label, query_label)
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
            view_num=4,
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
                  exp_mode=parameter_config['exp_mode']
                  )
    pred = mvccmodel.predict(query_norm_data, query_sm_arr, parameter_config['epoch_cpm_query'],
                             parameter_config['k_neighbor'])

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

    run.finish()
    return ret


def predict():
    model = torch.load('model/mvccmodel_' + data_config['query_key'] + ".pt")
    max_acc = 0
    query_data, query_label = read_data_label_h5(data_config['root_path'], data_config['query_key'])
    query_norm_data = mean_norm(query_data)
    query_sm_arr = [read_similarity_mat_h5(data_config['root_path'], data_config['query_key'] + "/sm_" + str(i + 1)) for
                    i in
                    range(4)]

    query_label = model.label_encoder.transform(query_label)
    for i in range(cycle):
        run = wandb.init(project="cell_classify_" + data_config['project'],
                         entity="kevislin",
                         config={"config": parameter_config, "data_config": data_config},
                         tags=[data_config['ref_name'] + '-' + data_config['query_name'], data_config['project']],
                         reinit=True)
        pred = model.predict(query_norm_data, query_sm_arr,
                             epoch_cpm_query=parameter_config['epoch_cpm_query'],
                             k_neighbor=parameter_config['k_neighbor'],
                             patience_for_cpm_query=parameter_config['patience_for_cpm_query'])

        # pred = model.label_encoder.inverse_transform(pred)
        # query_label = model.label_encoder.inverse_transform(query_label)
        acc = accuracy_score(query_label, pred)
        acc_arr.append(acc)
        if acc > max_acc:
            max_acc = acc

        run.finish()

    print("After {:} cycle, mean acc is {:.3f}, max acc is {:.3f}".format(cycle, sum(acc_arr) / len(acc_arr), max_acc))
    print(acc_arr)
    return max_acc


'''
    predict
'''
# final_acc = []
# data_config['query_key'] = 'query_1'
# final_acc.append(predict())
# print("--------------up-----------------")
# data_config['query_key'] = 'query_2'
# final_acc.append(predict())
# print("--------------up-----------------")
# data_config['query_key'] = 'query_3'
# final_acc.append(predict())
# print("---------------up----------------")
# data_config['query_key'] = 'query_4'
# final_acc.append(predict())
# print("----------------up---------------")
# data_config['query_key'] = 'query_5'
# final_acc.append(predict())
# print(final_acc)
# exit()

'''
train
'''
for i in range(cycle):
    ret = main_process()
    acc = accuracy_score(ret['pred'], ret['query_label'])
    acc_arr.append(acc)
    if acc > max_acc:
        max_acc = acc
        torch.save(ret['mvcc_model'], 'model/mvccmodel_' + data_config['query_key'] + ".pt")

print("After {:} cycle, mean acc is {:.3f}, max acc is {:.3f}".format(cycle, sum(acc_arr) / len(acc_arr), max_acc))
print(acc_arr)
