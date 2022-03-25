import os.path
# import pandas as pd
# import pandas as pd
# import sklearn.decomposition
import pandas as pd
import torch
from torch import nn
from utils import sc_normalization, mask_data, construct_graph, read_data_label, read_similarity_mat, \
    cpm_classify, z_score_normalization, show_cluster, concat_views
from model import scGNN, CPMNets
# from torch.utils.data import Dataset
import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import cluster
# import seaborn as sns
# import pandas as pd
# from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score
# import umap
# import scipy.io as spio
import wandb
# import sklearn.preprocessing as preprocess

def get_rid_of_0_gene(df1, df2):
    gene_idx = list(map(lambda x: x[0] & x[1], zip((df1.std() != 0).tolist(), (df2.std() != 0).tolist())))
    df1 = df1.iloc[:, gene_idx]
    df2 = df2.iloc[:, gene_idx]
    return df1, df2

#
#
#
# cel_seq_data = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/PBMC/CEL_Seq2/CEL_Seq2_data.csv', index_col=0)
#
# cel_seq_label = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/PBMC/CEL_Seq2/CEL_Seq2_label.csv')
# data_10x_v3 = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/PBMC/10X_V3/10X_v3_data.csv', index_col=0)
# label_10x_v3 = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/PBMC/10X_V3/10X_v3_label.csv')
#
# cell_type = ['Natural killer cell', 'Cytotoxic T cell', 'CD4+ T cell', 'B cell', 'CD14+ monocyte', 'Megakaryocyte', 'CD16+ monocyte']
#
# idx =label_10x_v3['CellType'].isin(cell_type).tolist()
# data_10x_v3 = data_10x_v3.iloc[idx, :]
# label_10x_v3 = label_10x_v3.iloc[idx, :]
#
# cel_seq_data, data_10x_v3 = get_rid_of_0_gene(cel_seq_data, data_10x_v3)
# cel_seq_label = cel_seq_label.replace(cell_type, [1,2,3,4,5,6,7])
# label_10x_v3 = label_10x_v3.replace(cell_type, [1,2,3,4,5,6,7])
#
# # gene_idx = list(map(lambda x: x[0] & x[1], zip((cel_seq_label.std() != 0).tolist(),(data_10x_v3.std()!=0).tolist())))
# # cel_seq_data = cel_seq_data.iloc[:, gene_idx]
# # data_10x_v3 = data_10x_v3.iloc[:, gene_idx]
#
# gene_names = cel_seq_data.columns.tolist()
# gene_names = [gene.split('_')[1] for gene in gene_names]
#
# cel_seq_data = pd.DataFrame(data=cel_seq_data.values, columns=gene_names, index=cel_seq_data.index.tolist())
# data_10x_v3 = pd.DataFrame(data=data_10x_v3.values, columns=gene_names, index=data_10x_v3.index.tolist())
#
# print(cel_seq_data.shape)
# print(cel_seq_label.shape)
# print(data_10x_v3.shape)
# print(label_10x_v3.shape)
#
#
# cel_seq_data.to_csv('cel_seq2_data.csv')
# cel_seq_label.to_csv('cel_seq2_label.csv', index=False)
#
# data_10x_v3.to_csv('10x_v3_data.csv')
# label_10x_v3.to_csv('10x_v3_label.csv', index=False)
#
# exit()


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
    l_arr = []
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
        l_arr.append(loss)
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print('Epoch: {}, Training Loss {:.4f}'.format(epoch, loss.item()))
    return model, l_arr


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
                    config['w_classify'])

    # 开始训练
    r_loss_arr = []
    c_loss_arr = []
    model.train_model(ref_data_embeddings, ref_label, config['epoch_CPM_train'], config['CPM_lr'], r_loss_arr,
                      c_loss_arr)

    # 对test_h进行adjust（按照论文的想法，保证consistency）
    test_loss_arr = []
    model.test(query_data_embeddings, config['epoch_CPM_test'], test_loss_arr)

    ref_h = model.get_h_train().detach().numpy()
    query_h = model.get_h_test().detach().numpy()
    cpm_loss = [r_loss_arr, c_loss_arr, test_loss_arr]
    return model, ref_h, query_h, cpm_loss


def transfer_label(data_path: dict,
                   label_path: dict,
                   sm_path: dict,
                   config: dict):
    '''
    :param data_path: 表达矩阵的路径
    :param label_path: 标签路径
    :param sm_path: 相似矩阵的路径
    :param config: 相关参数的配置
    :return:
    '''
    ref_data, ref_label = read_data_label(data_path['ref'], label_path['ref'])

    # 数据预处理
    ref_norm_data = sc_normalization(ref_data)
    masked_prob = min(len(ref_norm_data.nonzero()[0]) / (ref_norm_data.shape[0] * ref_norm_data.shape[1]), 0.3)
    masked_ref_data, index_pair, masking_idx = mask_data(ref_norm_data, masked_prob)


    ref_sm_arr = [read_similarity_mat(sm_path['ref'][0]),
                  read_similarity_mat(sm_path['ref'][1]),
                  read_similarity_mat(sm_path['ref'][2]),
                  read_similarity_mat(sm_path['ref'][3])]

    ref_graphs = [construct_graph(masked_ref_data, ref_sm_arr[0], config['k']),
                  construct_graph(masked_ref_data, ref_sm_arr[1], config['k']),
                  construct_graph(masked_ref_data, ref_sm_arr[2], config['k']),
                  construct_graph(masked_ref_data, ref_sm_arr[3], config['k'])]

    ref_views = []
    GNN_models = []
    GNN_loss_arr = []
    # 训练ref data in scGNN
    for i in range(len(ref_graphs)):
        model = scGNN(ref_graphs[i], config['middle_out'])
        optimizer = torch.optim.Adam(model.parameters(), lr=config['GNN_lr'])
        model, l_arr = train_scGNN(model, config['epoch_GCN'], ref_graphs[i], optimizer, index_pair, masking_idx,
                                   ref_norm_data, 'GNN: view'+str(i+1)+' loss')
        GNN_loss_arr.append(l_arr)
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
    ref_data_embeddings_tensor = torch.from_numpy(z_score_normalization(concat_views(ref_views))).float()
    ref_label_tensor = torch.from_numpy(ref_label).view(1, ref_label.shape[0]).long()

    '''
        Query data
    '''
    query_data, query_label = read_data_label(data_path['query'], label_path['query'])

    # 数据预处理
    query_norm_data = sc_normalization(query_data)


    # 构造Query data的Graph
    query_sm_arr = [read_similarity_mat(sm_path['query'][0]),
                    read_similarity_mat(sm_path['query'][1]),
                    read_similarity_mat(sm_path['query'][2]),
                    read_similarity_mat(sm_path['query'][3])]

    query_graphs = [construct_graph(query_norm_data, query_sm_arr[0], config['k']),
                    construct_graph(query_norm_data, query_sm_arr[1], config['k']),
                    construct_graph(query_norm_data, query_sm_arr[2], config['k']),
                    construct_graph(query_norm_data, query_sm_arr[3], config['k'])]

    # 获得Embedding
    query_views = []
    for i in range(len(GNN_models)):
        query_views.append(GNN_models[i].get_embedding(query_graphs[i]).detach().cpu().numpy())

    query_data_embeddings_tensor = torch.from_numpy(z_score_normalization(concat_views(query_views))).float()

    query_label_tensor = torch.from_numpy(query_label).view(1, query_label.shape[0]).long()

    '''
        CPM-Net
    '''
    cpm_model, ref_h, query_h, cpm_loss_arr = train_cpm_net(ref_data_embeddings_tensor,
                                                        ref_label_tensor,
                                                        query_data_embeddings_tensor,
                                                        ref_view_num,
                                                        ref_view_feat_len,
                                                        config)

    pred = cpm_classify(ref_h, query_h, ref_label)
    acc = (pred == query_label).sum()
    acc = acc / pred.shape[0]
    ret = {
        'acc': acc,
        'ref_h': ref_h,
        'query_h': query_h,
        'ref_raw_data': ref_data,
        'ref_label': ref_label,
        'query_raw_data': query_data,
        'query_label': query_label,
        'pred': pred,
        'gnn_loss': GNN_loss_arr,
        'cpm_loss': cpm_loss_arr
    }
    # print("Prediction Accuracy is {:.3f}".format(acc))
    return ret


# 数据路径
data_path = '/home/zhianhuang/yuanhuang/kevislin/data/transfer_across_platform_data/PBMC/cel_seq_10x_v3'

# 给出ref和query data所在的路径
dataPath = {
    'ref': os.path.join(data_path, 'cel_seq2_data.csv'),
    'query': os.path.join(data_path, '10x_v3_data.csv'),
}
# label所在的路径
labelPath = {
    'ref': os.path.join(data_path, 'cel_seq2_label.csv'),
    'query': os.path.join(data_path, '10x_v3_label.csv'),
}

sm_path = os.path.join(data_path, 'similarity_mat')
SMPath = {
    'ref': [
        os.path.join(sm_path, "SM_cel_seq_KEGG.csv"),
        os.path.join(sm_path, "SM_cel_seq_Reactome.csv"),
        os.path.join(sm_path, "SM_cel_seq_Wikipathways.csv"),
        os.path.join(sm_path, "SM_cel_seq_yan.csv"),
    ],
    'query': [
        os.path.join(sm_path, "SM_10x_v3_KEGG.csv"),
        os.path.join(sm_path, "SM_10x_v3_Reactome.csv"),
        os.path.join(sm_path, "SM_10x_v3_Wikipathways.csv"),
        os.path.join(sm_path, "SM_10x_v3_yan.csv"),
    ]
}

config = {
    'epoch_GCN': 6500,  # Huang model 训练的epoch
    'epoch_CPM_train': 4000,
    'epoch_CPM_test': 4000,
    'lsd_dim': 128,  # CPM_net latent space dimension
    'GNN_lr': 0.001,
    'CPM_lr': [0.001, 0.001],  # CPM_ner中train和test的学习率
    'ref_class_num': 9,  # Reference data的类别数
    'query_class_num': 9,  # query data的类别数
    'k': 4,  # 图构造的时候k_neighbor参数
    'middle_out': 256,  # GCN中间层维数
    'w_classify': 5  # classfication loss的权重
}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.init(project="Cell_Classification", entity="kevislin", config=config)

print("Transfer across platforms")
print("Reference: cel_seq", "Query: 10x_v3")


ret = transfer_label(dataPath, labelPath, SMPath, config)

# 结果打印
show_cluster(ret['ref_raw_data'], ret['ref_label'], 'Raw reference data')
show_cluster(ret['query_raw_data'], ret['query_label'], 'Raw query data')
show_cluster(ret['ref_h'], ret['ref_label'], 'Reference h')
show_cluster(ret['query_h'], ret['query_label'], 'Query h')
show_cluster(ret['query_h'], ret['pred'], 'Query h with prediction label')
show_cluster(np.concatenate([ret['ref_h'], ret['query_h']], axis=0), np.concatenate([ret['ref_label'], ret['pred']]),
             'Mouse-Human H distribution')

s_score = silhouette_score(ret['query_h'], ret['pred'])
ari = adjusted_rand_score(ret['query_label'], ret['pred'])
print("Prediction Accuracy is {:.3f}".format(ret['acc']))
print('Prediction Silhouette score is {:.3f}'.format(s_score))
print('Prediction ARI is {:.3f}'.format(ari))

# 数据上报
wandb.log({
    'Prediction Acc':ret['acc'],
    'Prediction Silhouette ': s_score,
    'ARI': ari
})

np.save(os.path.join(os.getcwd(), 'result'), ret['ref_h'])
np.save(os.path.join(os.getcwd(), 'result'), ret['query_h'])
