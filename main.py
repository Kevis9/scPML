import os.path
import torch
from torch import nn
from utils import Normalization, Mask_Data, Graph, readSCData, \
    setByPathway, readSimilarityMatrix, \
    Classify, z_score_Normalization, sharedGeneMatrix, showClusters
from Model import scGNN, CPMNets
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score

# 训练scGNN，得到每个Pathway的embedding
def train_scGNN_wrapper(model, n_epochs, G_data, optimizer, index_pair, masking_idx, scDataNorm):
    '''
    :param model: 待训练的模型
    :param n_epochs:
    :param G_data: 训练的图数据
    :param optimizer:
    :param index_pair: 做过mask元素的index
    :param masking_idx: mask元素的index
    :param scDataNorm: mask之前的normdata
    :return:
    '''
    model = model.to(device)
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(G_data.to(device))
        # 得到预测的droout
        dropout_pred = pred[index_pair[0][masking_idx], index_pair[1][masking_idx]]
        dropout_true = scDataNorm[index_pair[0][masking_idx], index_pair[1][masking_idx]]

        loss_fct = nn.MSELoss()
        loss = loss_fct(dropout_pred.view(1, -1), torch.tensor(dropout_true, dtype=torch.float).to(device).view(1, -1))

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print('Epoch: {}, Training Loss {:.4f}'.format(epoch, loss.item()))

    return model


def transfer_labels(dataPath, labelPath, SMPath, config):
    '''

    :param dataPath: 表达矩阵的路径
    :param labelPath: 标签路径
    :param SMPath: 相似矩阵的路径
    :param config: 相关参数的配置
    :return:
    '''
    ref_Data, ref_labels = readSCData(dataPath['ref'], labelPath['ref'])
    showClusters(ref_Data, ref_labels, 'raw reference data')

    # 数据预处理
    ref_norm_data = Normalization(ref_Data)
    # ref_norm_data = z_score_Normalization(ref_norm_data)

    masked_prob = min(len(ref_norm_data.nonzero()[0]) / (ref_norm_data.shape[0] * ref_norm_data.shape[1]), 0.3)
    masked_ref_data, index_pair, masking_idx = Mask_Data(ref_norm_data, masked_prob)
    showClusters(masked_ref_data, ref_labels, "ref masked data")

    ref_sm_arr = [readSimilarityMatrix(SMPath['ref'][0]),
                  readSimilarityMatrix(SMPath['ref'][1]),
                  readSimilarityMatrix(SMPath['ref'][2]),
                  readSimilarityMatrix(SMPath['ref'][3])]

    ref_graphs = [Graph(masked_ref_data, ref_sm_arr[0], config['k']),
                  Graph(masked_ref_data, ref_sm_arr[1], config['k']),
                  Graph(masked_ref_data, ref_sm_arr[2], config['k']),
                  Graph(masked_ref_data, ref_sm_arr[3], config['k'])]

    ref_views = []
    GCN_models = []

    # 训练ref data in scGNN
    for i in range(len(ref_graphs)):
        model = scGNN(ref_graphs[i], config['middle_out'])
        optimizer = torch.optim.Adam(model.parameters())
        model = train_scGNN_wrapper(model, config['epoch_GCN'], ref_graphs[i], optimizer, index_pair, masking_idx, ref_norm_data)
        model = model.to(device)
        # 利用未mask的矩阵，构造图，丢入训练好的model，得到中间层embedding
        # embedding = model.get_embedding(Graph(scDataNorm, similarity_matrix_arr[i]))
        # 还是用mask好的数据得到Embedding比较符合自监督的逻辑
        embedding = model.get_embedding(ref_graphs[i])
        ref_views.append(embedding.detach().cpu().numpy())
        GCN_models.append(model)

    '''
        Reference Data Embeddings  
    '''
    # view的个数
    ref_view_num = len(ref_views)
    # 每个view的特征长度可能不一样 (在这里是一样的)
    ref_view_feat = []
    for i in range(ref_view_num):
        ref_view_feat.append(ref_views[i].shape[1])
    # 查看所有view的类别分布情况
    for i in range(ref_view_num):
        showClusters(ref_views[i], ref_labels, 'ref view' + str(i + 1))

    # 把所有的view连接在一起
    ref_data_embeddings = np.concatenate(ref_views, axis=1).astype(np.float64)
    # 做一个z-score归一化
    # ref_data_embeddings = z_score_Normalization(ref_data_embeddings)
    ref_data_embeddings = torch.from_numpy(ref_data_embeddings).float()
    ref_label_tensor = torch.from_numpy(ref_labels).view(1, ref_labels.shape[0]).long()

    # 可视化reference data embedding
    showClusters(ref_data_embeddings, ref_labels, 'reference data embeddings')

    '''
        Query data
    '''
    query_scData, query_Label = readSCData(dataPath['query'],
                                           labelPath['query'])

    # 可视化原数据分布
    print(query_scData.shape)
    print(query_Label.shape)
    showClusters(query_scData, query_Label, 'Raw Query Data')

    # 数据预处理
    query_norm_scData = Normalization(query_scData)

    # 构造Query data的Graph
    query_sm_arr = [readSimilarityMatrix(SMPath['query'][0]),
                    readSimilarityMatrix(SMPath['query'][1]),
                    readSimilarityMatrix(SMPath['query'][2]),
                    readSimilarityMatrix(SMPath['query'][3])]

    query_graphs = [Graph(query_norm_scData, query_sm_arr[0], config['k']),
                    Graph(query_norm_scData, query_sm_arr[1], config['k']),
                    Graph(query_norm_scData, query_sm_arr[2], config['k']),
                    Graph(query_norm_scData, query_sm_arr[3], config['k'])]

    # 获得Embedding
    query_embeddings = []
    for i in range(len(GCN_models)):
        query_embeddings.append(GCN_models[i].get_embedding(query_graphs[i]).detach().cpu().numpy())
        showClusters(query_embeddings[i], query_Label, 'query view' + str(i + 1))

    query_data_embeddings = np.concatenate(query_embeddings, axis=1).astype(np.float64)
    # 做一个z-score归一化
    # query_data_embeddings = z_score_Normalization(query_data_embeddings)
    query_data_embeddings = torch.from_numpy(query_data_embeddings).float()
    query_label_tensor = torch.from_numpy(query_Label).view(1, query_Label.shape[0]).long()

    # 可视化query data embedding
    showClusters(query_data_embeddings, query_Label, 'query data embeddings')

    '''
        CPM-Net
    '''
    train_len = ref_data_embeddings.shape[0]
    test_len = query_data_embeddings.shape[0]

    # lsd_dim 作为超参数可调
    model = CPMNets(ref_view_num, train_len, test_len, ref_view_feat, class_num = 11, lsd_dim=config['lsd_dim'])

    n_epochs = config['epoch_CPM']

    # 开始训练
    model.train_model(ref_data_embeddings, ref_label_tensor, n_epochs, lr=config['CPM_lr'])

    # 对test_h进行adjust（按照论文的想法，保证consistency）
    model.test(query_data_embeddings, n_epochs)

    ref_h = model.get_h_train()
    query_h = model.get_h_test()

    # 保存ref_h
    ref_h = ref_h.detach().numpy()
    ref_h_path = os.path.join(os.getcwd(), "ref_h.npy")
    np.save(ref_h_path, ref_h)

    # 保存query_h
    query_h = query_h.detach().numpy()
    query_h_path = os.path.join(os.getcwd(), "query_h.npy")
    np.save(query_h_path, query_h)

    '''
        ref_h做一个 k-means聚类
    '''
    kmeans_model = cluster.KMeans(n_clusters=config['ref_class_num'], max_iter=500, init="k-means++", random_state=42)
    ref_h_labels = kmeans_model.fit_predict(ref_h)
    showClusters(ref_h, ref_h_labels, 'reference h')
    ref_s_score = silhouette_score(ref_h,ref_h_labels)
    ref_ari = adjusted_rand_score(ref_labels, ref_h_labels)
    print("Reference K-means result: Silhouette score is : {}, ARI is :{}".format(ref_s_score, ref_ari))

    '''
        query_h做一个 k-means聚类
    '''
    kmeans_model = cluster.KMeans(n_clusters=config['query_class_num'], max_iter=500, init="k-means++", random_state=42)
    q_h_labels = kmeans_model.fit_predict(query_h)
    showClusters(query_h, q_h_labels, 'query h')
    q_s_score = silhouette_score(query_h, q_h_labels)
    q_ari = adjusted_rand_score(query_Label, q_h_labels)
    print("Query K-means result: Silhouette score is : {}, ARI is :{}".format(q_s_score, q_ari))


dataset_name = "transfer_across_species_data"
data_path_pre = os.path.join(os.getcwd(), "..", dataset_name, "scData")
label_path_pre = os.path.join(os.getcwd(), "..", dataset_name, "label")

ref_data_name = "mouse_pancreas.csv"
ref_label_name = "mouse_pancreas_label.csv"

query_data_name = "human_pancreas.csv"
query_label_name = "human_pancreas_label.csv"

sm_path_pre = os.path.join(os.getcwd(), "..", dataset_name, "similarity_matrix")
ref_SM_path = os.path.join(sm_path_pre, "mouse")
query_SM_path = os.path.join(sm_path_pre, "human")


# 给出ref和query data所在的路径
dataPath = {
    'ref': os.path.join(data_path_pre, ref_data_name),
    'query': os.path.join(data_path_pre, query_data_name),
}
# label所在的路径
labelPath = {
    'ref': os.path.join(label_path_pre, ref_label_name),
    'query': os.path.join(label_path_pre, query_label_name),
}


SMPath = {
    'ref': [
        os.path.join(ref_SM_path, "SM_mouse_pancreas_KEGG.csv"),
        os.path.join(ref_SM_path, "SM_mouse_pancreas_Reactome.csv"),
        os.path.join(ref_SM_path, "SM_mouse_pancreas_Wikipathways.csv"),
        os.path.join(ref_SM_path, "SM_mouse_pancreas_biase.csv"),
    ],
    'query': [
        os.path.join(query_SM_path, "SM_human_pancreas_KEGG.csv"),
        os.path.join(query_SM_path, "SM_human_pancreas_Reactome.csv"),
        os.path.join(query_SM_path, "SM_human_pancreas_Wikipathways.csv"),
        os.path.join(query_SM_path, "SM_human_pancreas_yan.csv"),
    ]
}

config = {
    'epoch_GCN':2000, # Huang model 训练的epoch
    'epoch_CPM':5000,
    'lsd_dim':128, # CPM_net latent space dimension
    'CPM_lr':[0.0005, 0.0005], # CPM_ner中train和test的学习率
    'ref_class_num':9, # Reference data的类别数
    'query_class_num':9, # query data的类别数
    'k':2, # 图构造的时候k_neighbor参数
    'middle_out':256 # GCN中间层维数
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transfer_labels(dataPath, labelPath, SMPath, config)


# 最后进行一个分类
# label_pre = torch.from_numpy(Classify(train_H, test_H, train_labels)).view(1, -1).long()
#
# print("Prediction Accuracy: %.3f" % ((label_pre == test_labels).sum().flaot()/(test_len)))
