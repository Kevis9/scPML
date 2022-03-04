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
# 获得两个表达矩阵的公共基因
# sharedGeneMatrix(path1, path2)

# 生成mouse_vips的label
# df = pd.read_csv('./transfer_across_tissue/label/mouse_VISP_label.csv')
# labels = df['class'].tolist()
# print(set(labels))
# labels = pd.DataFrame(data=labels,columns=['class'])
# labels.to_csv('./mouse_VIPS_label.csv',index=False)
# exit()

# mouse_visp:随机取2000个数据和label
# mouse_visp_df = pd.read_csv('./transfer_across_tissue/scData/mouse_VISP.csv',index_col=0)
# labels_df = pd.read_csv('transfer_across_tissue/label/mouse_VISP_label.csv')
#
# num = labels_df.shape[0]
# idx = [i for i in range(num)]
# np.random.shuffle(idx)
#
#
# mouse_visp_df = mouse_visp_df.iloc[idx[:2000],:]
# labels_df = labels_df.iloc[idx[:2000],:]
# mouse_visp_df = pd.DataFrame(data=mouse_visp_df.values, columns=mouse_visp_df.columns)
#
# mouse_visp_df.to_csv('./mouse_VIPS_cut.csv')
# labels_df.to_csv('./mouse_VIPS_label_cut.csv', index=False)

# 对mouse_VISP做一个数据清理，把Label为Low Quality和No Class的样本去掉
# mouse_visp_df = pd.read_csv('./transfer_across_tissue/scData/mouse_VISP_cut.csv', index_col=0)
# label = pd.read_csv('./transfer_across_tissue/label/mouse_VIPS_label_cut.csv')
# idx = (label['class'] != 'No Class')&(label['class'] != 'Low Quality')
#
# mouse_visp_df = mouse_visp_df.loc[idx,:]
# mouse_visp_df = pd.DataFrame(data=mouse_visp_df.values, columns=mouse_visp_df.columns)
# label = label.loc[idx,:]
#
# mouse_visp_df.to_csv('./mouse_VISP_cut.csv')
# label.to_csv('./mouse_VISP_label_cut.csv', index=False)

# mouse_VISP_label_cut 数字化
# label = pd.read_csv('transfer_across_tissue/label/mouse_VISP_label_cut.csv')
# label_arr = list(set(label['class'].tolist()))
# for i in range(len(label_arr)):
#     label = label.replace(label_arr[i], i+1)
#
# label.to_csv('./mouse_VISP_label_cut_num.csv', index=False)

# 读取mouse_pancreas的label并且数字化
# label = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/GSE84133_RAW/GSM2230761_mouse1_umifm_counts.csv')
# label = pd.DataFrame(data=label['assigned_cluster'], columns=['assigned_cluster'])
#
# label_arr = list(set(label['assigned_cluster'].tolist()))
# print(len(label_arr))
# for i in range(len(label_arr)):
#     label = label.replace(label_arr[i], i+1)
# label.to_csv('./mouse1_pancreas_label.csv', index=False)
# exit()

# 获取老鼠的基因
# df = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/GSE84133_RAW/GSM2230762_mouse2_umifm_counts.csv', index_col=0)
# del df['barcode']
# del df['assigned_cluster']
# gene_names = df.columns.tolist()
# mouse_gene_names = pd.DataFrame(data=gene_names, columns=['mouse_gene_names'])
# mouse_gene_names.to_csv('./mouse_gene_names.csv', index=False)
# exit()
#
# '''
#     GSE84133:生成mouse2和human3的公共基因矩阵
#     并且label数字化
# '''
# mouse1_df = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/GSE84133_RAW/GSM2230761_mouse1_umifm_counts.csv', index_col=0)
# mouse2_df = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/GSE84133_RAW/GSM2230762_mouse2_umifm_counts.csv', index_col=0)
#
# human1_df = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/GSE84133_RAW/GSM2230757_human1_umifm_counts.csv', index_col=0)
# human2_df = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/GSE84133_RAW/GSM2230758_human2_umifm_counts.csv', index_col=0)
# human3_df = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/GSE84133_RAW/GSM2230759_human3_umifm_counts.csv', index_col=0)
# human4_df = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/GSE84133_RAW/GSM2230760_human4_umifm_counts.csv', index_col=0)
#
#
# common_gene_names = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/transfer_across_species_data/gene_names/commom_gene_names.csv', index_col=0)
#
# # 把数据纵向连接起来
# human_df = pd.concat([human1_df, human2_df, human3_df, human4_df])
# mouse_df = pd.concat([mouse1_df, mouse2_df])
#
# # 查看下各个class的出现次数 (突然觉得pandas内置的函数用的好爽）
# # human_label_counts = human_df.loc[:,"assigned_cluster"].value_counts()
# # mouse_label_counts = mouse_df.loc[:,"assigned_cluster"].value_counts()
#
# cell_type_arr = ['beta', 'alpha', 'ductal', 'acinar', 'delta', 'activated_stellate', 'gamma', 'endothelial', 'quiescent_stellate', 'macrophage', 'mast']
# cell_type_arr.sort()
#
# # 属于cell_type的样本
# human_df = human_df.loc[human_df['assigned_cluster'].isin(cell_type_arr), :]
# mouse_df = mouse_df.loc[mouse_df['assigned_cluster'].isin(cell_type_arr), :]
#
#
# label_num = [i+1 for i in range(len(cell_type_arr))]
# mouse_df['assigned_cluster'].replace(cell_type_arr, label_num, inplace=True) # inplace代表修改原来的df
# human_df['assigned_cluster'].replace(cell_type_arr, label_num, inplace=True)

# for i in range(len(cell_type_arr)):
#     mouse_df['assigned_cluster'].replace(cell_type_arr[i], i+1)
#     human_df['assigned_cluster'].replace(cell_type_arr[i], i+1)
#     label_dic[cell_type_arr[i]] = i+1


# mouse_label_df = pd.DataFrame(data=mouse_df['assigned_cluster'].tolist(), columns=['class'])
# human_label_df = pd.DataFrame(data=human_df['assigned_cluster'].tolist(), columns=['class'])
#
#
# # GSE84133: 生成label, 并进行数字化
# mouse_label_df.to_csv('./mouse_pancreas_label.csv', index=False)
# human_label_df.to_csv('./human_pancreas_label.csv', index=False)
#
#
# # 对原表达矩阵基因和同源基因再取交集
# human_names = human_df.columns.intersection(common_gene_names['human']).tolist()
# idx = common_gene_names['human'].isin(human_names)
#
# common_gene_names = common_gene_names.loc[idx, :]
#
# mouse_df = mouse_df[common_gene_names['mouse']]
# human_df = human_df[common_gene_names['human']]
#
# mouse_df.to_csv('./mouse_pancreas.csv')
# human_df.to_csv('./human_pancreas.csv')
#
# print(mouse_df.shape)
# print(human_df.shape)
# exit()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Reference数据读取, 得到表达矩阵和标签（Reference）
dataPath = os.path.join(os.getcwd(), "..", "transfer_across_species_data")
scData, scLabels = readSCData(os.path.join(dataPath, "scData", "mouse_pancreas.csv"), os.path.join(dataPath, "label", "mouse_pancreas_label.csv"))

showClusters(scData, scLabels, 'Raw Reference Data')

'''
    数据预处理
'''
# 对单细胞表达矩阵做归一化
scDataNorm = Normalization(scData)
# scDataNorm = z_score_Normalization(scDataNorm)

#对数据进行随机mask
masked_prob = min(len(scDataNorm.nonzero()[0]) / (scDataNorm.shape[0] * scDataNorm.shape[1]), 0.3)
# 得到被masked之后的数据
masked_data, index_pair, masking_idx = Mask_Data(scDataNorm, masked_prob)

# 对mask数据进行一个可视化
showClusters(masked_data, scLabels, "Ref: masked data")

'''
    由Similarity matrix，构造出Graph，每个节点的值是表达矩阵的feature        
'''
matrix_path = os.path.join(os.getcwd(), "..", "transfer_across_species_data", "similarity_matrix", "mouse")
similarity_matrix_arr = [readSimilarityMatrix(os.path.join(matrix_path, 'SM_mouse_pancreas_KEGG.csv')),
                         readSimilarityMatrix(os.path.join(matrix_path, 'SM_mouse_pancreas_Reactome.csv')),
                         readSimilarityMatrix(os.path.join(matrix_path, 'SM_mouse_pancreas_Wikipathways.csv')),
                         readSimilarityMatrix(os.path.join(matrix_path, 'SM_mouse_pancreas_biase_150.csv'))]


graphs = [Graph(masked_data, similarity_matrix_arr[0]),
          Graph(masked_data, similarity_matrix_arr[1]),
          Graph(masked_data, similarity_matrix_arr[2]),
          Graph(masked_data, similarity_matrix_arr[3])]


'''
    训练scGNN，得到每个Pathway的embedding
'''
def train_scGNN_wrapper(model, n_epochs, G_data, optimizer):

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


views = []
models = []
n_epochs = 1500

# 训练
for i in range(len(graphs)):
    model = scGNN(graphs[i])
    optimizer = torch.optim.Adam(model.parameters())
    model = train_scGNN_wrapper(model, n_epochs, graphs[i], optimizer)
    model = model.to(device)
    # 利用未mask的矩阵，构造图，丢入训练好的model，得到中间层embedding
    # embedding = model.get_embedding(Graph(scDataNorm, similarity_matrix_arr[i]))
    # 还是用mask好的数据得到Embedding比较符合自监督的逻辑
    embedding = model.get_embedding(graphs[i])
    views.append(embedding.detach().cpu().numpy())
    models.append(model)


'''
    Reference Data Embeddings  
'''

# view的个数
view_num = len(views)

# 每个view的特征长度可能不一样 (在这里是一样的)
view_feat = []
for i in range(view_num):
    view_feat.append(views[i].shape[1])


# 查看所有view的类别分布情况
for i in range(view_num):
    showClusters(views[i], scLabels, 'Ref: view'+str(i+1))


# 把所有的view连接在一起
ref_data_embeddings = np.concatenate(views, axis=1).astype(np.float64)
# 做一个z-score归一化
ref_data_embeddings = z_score_Normalization(ref_data_embeddings)
ref_data_embeddings = torch.from_numpy(ref_data_embeddings).float()
ref_label_tensor = torch.from_numpy(scLabels).view(1, scLabels.shape[0]).long()

# 可视化reference data embedding
showClusters(ref_data_embeddings, scLabels, 'reference data embeddings')


'''
    Query data
'''
query_scData, query_Label = readSCData(os.path.join(dataPath, "scData", "mouse_VISP_cut.csv"), os.path.join(dataPath, "label","mouse_VISP_label_cut.csv"))

# 可视化原数据分布
showClusters(query_scData, query_Label, 'Raw Query Data')

# 数据预处理
query_norm_scData = Normalization(query_scData)

# 构造Query data的Graph
matrix_path = os.path.join(os.getcwd(), "..", "transfer_across_species_data", "similarity_matrix", "human")
similarity_matrix_arr = [readSimilarityMatrix(os.path.join(matrix_path, 'SM_human_pancreas_KEGG.csv')),
                         readSimilarityMatrix(os.path.join(matrix_path, 'SM_human_pancreas_Reactome.csv')),
                         readSimilarityMatrix(os.path.join(matrix_path, 'SM_human_pancreas_Wikipathways.csv')),
                         readSimilarityMatrix(os.path.join(matrix_path, 'SM_human_pancreas_biase_150.csv'))]

query_graphs = [Graph(query_norm_scData, similarity_matrix_arr[0]),
                Graph(query_norm_scData, similarity_matrix_arr[1]),
                Graph(query_norm_scData, similarity_matrix_arr[2]),
                Graph(query_norm_scData, similarity_matrix_arr[3])]

# 获得Embedding
query_embeddings = []
for i in range(len(models)):
    query_embeddings.append(models[i].get_embedding(query_graphs[i]).detach().cpu().numpy())
    showClusters(query_embeddings[i], query_Label, 'query: view'+str(i+1))


query_data_embeddings = np.concatenate(query_embeddings, axis=1).astype(np.float64)
# 做一个z-score归一化
query_data_embeddings = z_score_Normalization(query_data_embeddings)
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
model = CPMNets(view_num, train_len, test_len, view_feat, lsd_dim=64)


n_epochs = 5000

# 开始训练
model.train_model(ref_data_embeddings, ref_label_tensor, n_epochs, lr=[0.0005, 0.0005])

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
kmeans_model = cluster.KMeans(n_clusters=10, max_iter=500, init="k-means++")
ref_h_labels = kmeans_model.fit_predict(ref_h)
showClusters(ref_h, ref_h_labels, 'reference h')

'''
    query_h做一个 k-means聚类
'''
kmeans_model = cluster.KMeans(n_clusters=10, max_iter=500, init="k-means++")
q_h_labels = kmeans_model.fit_predict(query_h)
showClusters(query_h, q_h_labels, 'query h')

# 数据可视化
# 利用t-sne降维
# tsne = TSNE()
# train_h_2d = tsne.fit_transform(train_h)
# plt.scatter(train_h_2d[:,0], train_h_2d[:, 1],c=model.labels_)
# plt.title('Train_h_kmeans')
# plt.show()

# 最后进行一个分类
# label_pre = torch.from_numpy(Classify(train_H, test_H, train_labels)).view(1, -1).long()
#
# print("Prediction Accuracy: %.3f" % ((label_pre == test_labels).sum().flaot()/(test_len)))