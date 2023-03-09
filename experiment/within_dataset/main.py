import sys

import sklearn.preprocessing
import torch

from MVCC.classifiers import FCClassifier

sys.path.append('../../../..')
import os

os.system("wandb disabled")
from MVCC.util import mean_norm, construct_graph_with_knn, \
    read_data_label_h5, read_similarity_mat_h5, encode_label, show_result, pre_process, z_score_scale, construct_graph, \
    get_similarity_matrix, check_out_similarity_matrix
from MVCC.model import MVCCModel
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import wandb

# 数据配置
data_config = {
    'root_path': '.',  # data.h5 path
    'ref_name': 'cel_seq2',
    'query_name': 'dropseq',
    'ref_key': 'query_1',  # PBMC2中ref3代替ref1，也就是10xv3（ref3）做ref1，seq well(ref1)做ref3
    'query_key': 'ref_1',
    'project': 'PBMC2',
}

projects = [  # '10x_v3',
    # 'cel_seq',
    # 'dropseq',
    # 'emtab5061',
    # 'gse81608',
    # 'gse84133_human',
    # 'gse84133_mouse',
    # 'gse85241',
    # 'indrop',
    # 'seq_well',
    # 'smart_seq'
    'GSE99254'
]

parameter_config = {
    'gcn_middle_out': 1024,  # GCN中间层维数
    'lsd': 512,  # CPM_net latent space dimension
    'lamb': 1,  # classfication loss的权重
    'epoch_cpm_ref': 500,
    'epoch_cpm_query': 50,
    'exp_mode': 1,  # 1: start from scratch,
                    # 2: multi ref ,
                    # 3: gcn model exists, train cpm model and classifier
    'nf': 2000,
    'classifier_name': "FC",
    # 不太重要参数
    'batch_size_classifier': 256,  # CPM中重构和分类的batch size
    'epoch_gcn': 200,  # Huang gcn 训练的epoch
    'epoch_classifier': 500,
    'patience_for_classifier': 20,
    'patience_for_gcn': 20,  # 训练GCN的时候加入一个早停机制
    'patience_for_cpm_ref': 50,  # cpm train ref 早停patience
    'patience_for_cpm_query': 50,  # query h 早停patience
    'k_neighbor': 0,  # GCN 图构造的时候k_neighbor参数
    'mask_rate': 0.3,
    'gamma': 1,
    'test_size': 0.2,
    'show_result': True,
    'view_num': 1,
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
    # ref_norm_data = ref_data
    # query_norm_data = query_data
    # np.savetxt("ref_data.csv", ref_norm_data, delimiter=',')
    # np.savetxt("query_data.csv", query_norm_data, delimiter=',')

    # ref_norm_data = sc_normalization(ref_data)
    # query_norm_data = sc_normalization(query_data)

    ref_sm_arr = [read_similarity_mat_h5(data_config['root_path'], data_config['ref_key'] + "/sm_" + str(i + 1)) for i
                  in
                  range(parameter_config['view_num'])]

    query_sm_arr = [read_similarity_mat_h5(data_config['root_path'], data_config['query_key'] + "/sm_" + str(i + 1)) for
                    i in
                    range(parameter_config['view_num'])]

    # check_out_similarity_matrix(ref_sm_arr[0], ref_label, parameter_config['k_neighbor'])
    # exit()
    # ref_sm_arr = [get_similarity_matrix(ref_norm_data, 3).A]
    # query_sm_arr = [get_similarity_matrix(query_norm_data, 3).A]


    if parameter_config['exp_mode'] == 2:
        # multi ref
        # mvccmodel = torch.load('model/mvccmodel_'+data_config['query_key']+".pt")
        mvccmodel = torch.load('model/mvccmodel_multi.pt')
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
    cpm_acc = (pred_cpm == query_label).sum() / pred_cpm.shape[0]
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

    ''' 单纯的FC '''
    classifier = FCClassifier(parameter_config['nf'], len(set(ref_label)))
    classifier = classifier.to(device='cuda:0')
    ref_label, query_label, enc = encode_label(ref_label, query_label)
    classifier.train_classifier(ref_norm_data,
                                ref_label,
                                100,
                                save_path='.',
                                test_size=0.2,
                                batch_size=128,
                                epochs=500,
                                lr=1e-3
                                )

    classifier.eval()
    with torch.no_grad():
        logits = classifier(torch.from_numpy(query_norm_data).float().to("cuda:0"))
        pred = logits.argmax(dim=1)
    pred = pred.cpu().detach().numpy()
    print("单纯 FC :{:.3f}".format(accuracy_score(pred, query_label)))

    ''' 
        ===========================================================
    '''

    '''
            GCN + FC 
        '''
    classifier = FCClassifier(parameter_config['gcn_middle_out'], len(set(ref_label)))
    classifier = classifier.to(device='cuda:0')
    ref_label, query_label, enc = encode_label(ref_label, query_label)

    ref_graph_data = construct_graph(ref_norm_data, ref_sm_arr[0], 0)

    ref_norm_data = z_score_scale(mvccmodel.gcn_models[0].get_embedding(ref_graph_data).detach().cpu().numpy())
    query_graph_data = construct_graph(query_norm_data, query_sm_arr[0], 0)
    query_norm_data = z_score_scale(mvccmodel.gcn_models[0].get_embedding(query_graph_data).detach().cpu().numpy())

    classifier.train_classifier(ref_norm_data,
                                ref_label,
                                100,
                                save_path='.',
                                test_size=0.2,
                                batch_size=128,
                                epochs=500,
                                lr=1e-3
                                )

    classifier.eval()
    with torch.no_grad():
        logits = classifier(torch.from_numpy(query_norm_data).float().to("cuda:0"))
        pred = logits.argmax(dim=1)
    pred = pred.cpu().detach().numpy()
    print("GCN + FC :{:.3f}".format(accuracy_score(pred, query_label)))

    ''' ================================================= '''

    return ret


# ret = main_process()
# acc = accuracy_score(ret['pred'], ret['query_label'])
# print("pred acc is {:.3f}".format(acc))


for i in range(len(projects)):
    data_config['root_path'] = projects[i]
    ret = main_process()
    # save model
    torch.save(ret['mvcc_model'],
               os.path.join(data_config['root_path'], 'model', 'mvccmodel_' + data_config['ref_key'] + ".pt"))
    torch.save(ret['mvcc_model'], os.path.join(data_config['root_path'], 'model', 'mvccmodel_multi.pt'))

    # acc = accuracy_score(ret['pred'], ret['query_label'])
    # sys.stdout = open("output_log.txt", "a")
    # print("{:}, {:.3f}".format(project[i], acc))
    # final_acc.append(acc)

# print(final_acc)
# final_acc = pd.DataFrame({
#     'project':project,
#     'acc':final_acc
# })

# final_acc.to_csv("final_acc.csv")

