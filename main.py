import os.path
import torch
from torch import embedding, nn
from utils import sc_normalization, mask_data, construct_graph, \
    read_data_label_h5, read_similarity_mat, \
    cpm_classify, z_score_normalization, show_cluster, \
    concat_views, batch_mixing_entropy, runPCA, runUMAP
from model import scGNN, CPMNets, Classifier
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score
import wandb
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset, ConcatDataset
# seq_well_smart 只有五类!!!
# drop_seq_10x_v3有8类

# 训练scGNN，得到每个Pathway的embedding
def train_scGNN(model, n_epochs, G_data, optimizer,
                index_pair, masking_idx, norm_data, loss_title):
    '''
    :param model: 待训练的模型
    :param n_epochs:
    :param G_data: 训练的图数据
    :param optimizer:
    :param index_pair: 做过mask元素的index pair
    :param masking_idx: mask元素的index
    :param norm_data: mask之后的norm_data
    :return:
    '''
    model = model.to(device)
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(G_data.to(device))

        # 得到预测的droout
        dropout_pred = pred[index_pair[0][masking_idx], index_pair[1][masking_idx]]
        dropout_true = norm_data[index_pair[0][masking_idx], index_pair[1][masking_idx]]
        loss_fct = nn.MSELoss()
        loss = loss_fct(dropout_pred.view(1, -1), torch.tensor(dropout_true, dtype=torch.float).to(device).view(1, -1))
        wandb.log({
            loss_title: loss.item()
        })
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print('Epoch: {}, Training Loss {:.4f}'.format(epoch, loss.item()))
    return model


def train_cpm_net(ref_data_embeddings: torch.Tensor,
                  ref_label: torch.Tensor,
                  query_data_embeddings: torch.Tensor,
                  ref_view_num: int,
                  ref_view_feat_len: list,
                  config: dict):
    train_len = ref_data_embeddings.shape[0]
    test_len = query_data_embeddings.shape[0]
    # lsd_dim 作为超参数可调
    model = CPMNets(ref_view_num, train_len, test_len, ref_view_feat_len, config['ref_class_num'], config['lsd_dim'],
                    config)

    # 开始训练
    model.train_ref_h(ref_data_embeddings, ref_label, config['epoch_CPM_train'], config['CPM_lr'])

    # 对test_h进行adjust（按照论文的想法，保证consistency）
    model.get_query_h(query_data_embeddings, config['epoch_CPM_test'])

    ref_h = model.get_h_train().detach().cpu().numpy()
    query_h = model.get_h_test().detach().cpu().numpy()
    
    return model, ref_h, query_h

def transfer_train(data_config: dict,
                   sm_path: dict,
                   config: dict):
    '''
    :param data_path: 表达矩阵的路径
    :param label_path: 标签路径
    :param sm_path: 相似矩阵的路径
    :param config: 相关参数的配置
    :return:
    '''
    data_path = os.path.join(data_config['data_path'], 'data.h5')

    ref_data, ref_label = read_data_label_h5(data_path, data_config['ref_name'])
    
    ref_data = ref_data.astype(np.float64)
    ref_enc = preprocessing.LabelEncoder()
    # label从1开始
    ref_label = (ref_enc.fit_transform(ref_label)).astype(np.int64)

    # 数据预处理
    ref_norm_data = sc_normalization(ref_data)
    # ref_norm_data = preprocessing.minmax_scale(ref_norm_data, axis=0)

    masked_prob = min(len(ref_norm_data.nonzero()[0]) / (ref_norm_data.shape[0] * ref_norm_data.shape[1]), 0.3)
    masked_ref_data, index_pair, masking_idx = mask_data(ref_norm_data, masked_prob)

    ref_sm_arr = [read_similarity_mat(sm_path['ref'][i]) for i in range(len(sm_path['ref']))]
    ref_graphs = [construct_graph(masked_ref_data, ref_sm_arr[i], config['k']) for i in range(len(ref_sm_arr))]

    ref_views = []
    GNN_models = []
    # 训练ref data in scGNN
    for i in range(len(ref_graphs)):
        model = scGNN(ref_graphs[i], config['middle_out'])
        optimizer = torch.optim.Adam(model.parameters(), lr=config['GNN_lr'])
        model = train_scGNN(model, config['epoch_GCN'], ref_graphs[i], optimizer, index_pair, masking_idx,
                            ref_norm_data, 'GNN: view' + str(i + 1) + ' loss')

        # 利用未mask的矩阵，构造图，丢入训练好的model，得到中间层embedding
        # embedding = model.get_embedding(Graph(scDataNorm, similarity_matrix_arr[i]))
        # 还是用mask好的数据得到Embedding比较符合自监督的逻辑
        embedding = model.get_embedding(ref_graphs[i])
        ref_views.append(embedding.detach().cpu().numpy())
        GNN_models.append(model)

    '''
        Reference Data Embeddings  
    '''
    # view的个数
    ref_view_num = len(ref_views)
    # 每个view的特征长度
    ref_view_feat_len = []
    for i in range(ref_view_num):
        ref_view_feat_len.append(ref_views[i].shape[1])

    # 把所有的view连接在一起
    ref_data_embeddings_tensor = torch.from_numpy(z_score_normalization(concat_views(ref_views))).float().to(device)
    ref_label_tensor = torch.from_numpy(ref_label).view(1, ref_label.shape[0]).long().to(device)

    '''
        Query data
    '''
    query_data, query_label = read_data_label_h5(data_path, data_config['query_name'])
    query_data = query_data.astype(np.float64)
    query_enc = preprocessing.LabelEncoder()
    query_label = (query_enc.fit_transform(query_label)).astype(np.int64)

    # 数据预处理
    query_norm_data = sc_normalization(query_data)
    # query_norm_data = preprocessing.minmax_scale(query_norm_data, axis=0)

    # 构造Query data的Graph
    query_sm_arr = [read_similarity_mat(sm_path['query'][i]) for i in range(len(sm_path['query']))]
    query_graphs = [construct_graph(query_norm_data, query_sm_arr[i], config['k']) for i in range(len(query_sm_arr))]

    # 获得Embedding
    query_views = []
    for i in range(len(GNN_models)):
        query_views.append(GNN_models[i].get_embedding(query_graphs[i]).detach().cpu().numpy())

    query_data_embeddings_tensor = torch.from_numpy(z_score_normalization(concat_views(query_views))).float().to(device)

    query_label_tensor = torch.from_numpy(query_label).view(1, query_label.shape[0]).long().to(device)

    '''
        CPM-Net
    '''
    cpm_model, ref_h, query_h = train_cpm_net(ref_data_embeddings_tensor,
                                              ref_label_tensor,
                                              query_data_embeddings_tensor,
                                              ref_view_num,
                                              ref_view_feat_len,
                                              config)


    pred = cpm_classify(ref_h, query_h, ref_label)
    acc = (pred == query_label).sum()
    acc = acc / pred.shape[0]

    # 还原label
    ref_label = ref_enc.inverse_transform(ref_label)

    query_label = query_enc.inverse_transform(query_label)

    pred = query_enc.inverse_transform(pred)

    ret = {
        'acc': acc,
        'ref_h': ref_h,
        'query_h': query_h,
        'ref_raw_data': ref_data,
        'ref_label': ref_label,
        'query_raw_data': query_data,
        'query_label': query_label,
        'pred': pred
    }
    return ret


# 数据配置
data_config = {
    'data_path': '/home/zhianhuang/yuanhuang/kevislin/data/species_data/GSE84133/no_diff',
    'ref_name': 'human',
    'query_name': 'mouse',
    'project': 'species',
    'class_num': 8,
    'dataset_name':'GSE84133'
}
#{'gamma', 'alpha', 'endothelial', 'macrophage', 'ductal', 'delta', 'beta', 'quiescent_stellate'}

config = {
    'epoch_GCN': 2000,  # Huang model 训练的epoch
    'epoch_CPM_train': 3000,
    'epoch_CPM_test': 3000,
    'lsd_dim': 128,  # CPM_net latent space dimension
    'GNN_lr': 0.001,
    'CPM_lr': [0.001, 0.001, 0.001],  # CPM_ner中net和train_h,test_h的学习率
    'ref_class_num': data_config['class_num'],  # Reference data的类别数
    'query_class_num': data_config['class_num'],  # query data的类别数
    'k': 2,  # 图构造的时候k_neighbor参数
    'middle_out': 4000,  # GCN中间层维数
    'w_classify': 1,  # classfication loss的权重
    'cen_weight': 0.5,
}

sm_path = os.path.join(data_config['data_path'], 'similarity_mat')
SMPath = {
    'ref': [
        os.path.join(sm_path, "SM_" + data_config['ref_name'] + "_KEGG.csv"),
        os.path.join(sm_path, "SM_" + data_config['ref_name'] + "_Reactome.csv"),
        os.path.join(sm_path, "SM_" + data_config['ref_name'] + "_Wikipathways.csv"),
        os.path.join(sm_path, "SM_" + data_config['ref_name'] + "_yan.csv"),
    ],
    'query': [
        os.path.join(sm_path, "SM_" + data_config['query_name'] + "_KEGG.csv"),
        os.path.join(sm_path, "SM_" + data_config['query_name'] + "_Reactome.csv"),
        os.path.join(sm_path, "SM_" + data_config['query_name'] + "_Wikipathways.csv"),
        os.path.join(sm_path, "SM_" + data_config['query_name'] + "_biase.csv"),
    ]
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.init(project="cell_classify_" + data_config['project'],
           entity="kevislin",
           config={"config":config, "data_config": data_config},
           tags=[data_config['ref_name'] + '-' + data_config['query_name'], data_config['project'],
                 str(data_config['class_num']) + ' class', data_config['dataset_name']])


print("Transfer across " + data_config['project'])
print("Reference: " + data_config['ref_name'], "Query: " + data_config['query_name'])

# 获取结果
ret = transfer_train(data_config, SMPath, config)
embedding_h = np.concatenate([ret['ref_h'], ret['query_h']], axis=0)
# embedding_h_pca = runPCA(embedding_h)

'''
    ==================结果处理====================
'''

ref_h = embedding_h[:ret['ref_h'].shape[0], :]
query_h = embedding_h[ret['ref_h'].shape[0]:, :]

# evaluation metrics
total_s_score = silhouette_score(embedding_h, list(ret['ref_label']) + list(ret['pred']))
ref_s_score = silhouette_score(ref_h, ret['ref_label'])
q_s_score = silhouette_score(query_h, ret['pred'])

ari = adjusted_rand_score(ret['query_label'], ret['pred'])
bme = batch_mixing_entropy(ref_h, query_h)
bme = sum(bme) / len(bme)

print("Prediction Accuracy is {:.3f}".format(ret['acc']))
# print('Prediction Silhouette score is {:.3f}'.format(s_score))
print('Prediction ARI is {:.3f}'.format(ari))

# 数据上报
wandb.log({
    'Prediction Acc': ret['acc'],
    'ref Silhouette ': ref_s_score,
    'query Silhouette ': q_s_score,
    'total Silhouette ': total_s_score,
    'ARI': ari,
    'Batch Mixing Entropy Mean': bme
})

raw_data = np.concatenate([ret['ref_raw_data'], ret['query_raw_data']], axis=0)
# raw_data_pca = runPCA(raw_data)
#
raw_data_2d = runUMAP(raw_data) # 对PCA之后的数据进行UMAP可视化
ref_len = ret['ref_raw_data'].shape[0]
#
h_data_2d = runUMAP(embedding_h)

np.save('result/raw_data_2d.npy', raw_data_2d)
np.save('result/h_data_2d.npy', h_data_2d)

all_true_labels = np.concatenate([ret['ref_label'], ret['query_label']]).reshape(-1)
all_pred_labels = np.concatenate([ret['ref_label'], ret['pred']]).reshape(-1)
np.save('result/all_true_labels.npy', all_true_labels)
np.save('result/all_pred_labels.npy', all_pred_labels)

show_cluster(raw_data_2d, all_true_labels, 'reference-query raw true label')
show_cluster(h_data_2d, all_true_labels, 'reference-query h true label')
show_cluster(h_data_2d, all_pred_labels, 'reference-query h pred label')

# show_cluster(raw_data_2d[:ref_len, :], ret['ref_label'], 'Raw reference data')
# show_cluster(raw_data_2d[ref_len:, :], ret['query_label'], 'Raw query data')
# show_cluster(h_data_2d[:ref_len, :], ret['ref_label'], 'Reference h')
# show_cluster(h_data_2d[ref_len:, :], ret['query_label'], 'Query h')
# show_cluster(h_data_2d[ref_len:, :], ret['pred'], 'Query h with prediction label')

# For multi omics part
# show_cluster(h_data_2d, np.concatenate([['Reference' for i in range(len(ret['ref_label']))], ['Query' for i in range(len(ret['query_label']))]])
#              , 'Reference-Query H')
# show_cluster(h_data_2d, np.concatenate([ret['ref_label'].reshape(-1), ret['query_label'].reshape(-1)])
#              , 'Reference-Query H with pred label')
# show_cluster(raw_data_2d, np.concatenate([['Reference' for i in range(len(ret['ref_label']))], ['Query' for i in range(len(ret['query_label']))]])
#              , 'Reference-Query Raw')
