import os.path

import torch
from torch import nn
from utils import Normalization, Mask_Data, Graph, readSCData, \
    setByPathway, readSimilarityMatrix, \
    Classify, z_score_Normalization, sharedGeneMatrix
from Model import scGNN, CPMNets
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# 数据读取, 得到单细胞表达矩阵和标签
dataPath = os.path.join(os.getcwd(), "..", "transfer_across_tissue_data")
scData, scLabels = readSCData(os.path.join(dataPath, "scData", "mouse1_pancreas.csv"), os.path.join(dataPath, "label","mouse1_pancreas_label.csv"))

# 对单细胞表达矩阵做归一化
scDataNorm = Normalization(scData)

'''
    对数据进行随机mask (仅仅模拟Dropout event)
'''
# 概率
masked_prob = min(len(scDataNorm.nonzero()[0]) / (scDataNorm.shape[0] * scDataNorm.shape[1]), 0.3)

# 得到被masked之后的数据
masked_data, index_pair, masking_idx = Mask_Data(scDataNorm, masked_prob)

tsne = TSNE()
masked_data_2d = tsne.fit_transform(masked_data)
plt.scatter(masked_data_2d[:, 0], masked_data_2d[:, 1], c=scLabels)
plt.title('masked data')
plt.show()

'''
    根据Cell Similarity矩阵，构造出Graph来，每个节点的feature是被masked之后的矩阵        
'''
matrix_path = os.path.join(os.getcwd(), "..", "transfer_across_tissue_data", "similarity_matrix", "mouse1")
similarity_matrix_arr = [readSimilarityMatrix(os.path.join(matrix_path, 'SM_mouse1_KEGG.csv')),
                         readSimilarityMatrix(os.path.join(matrix_path, 'SM_mouse1_Reactome.csv')),
                         readSimilarityMatrix(os.path.join(matrix_path, 'SM_mouse1_Wikipathways.csv')),
                         readSimilarityMatrix(os.path.join(matrix_path, 'SM_mouse1_de_novo_pathway.csv'))]


graphs = [Graph(masked_data, similarity_matrix_arr[0]),
          Graph(masked_data, similarity_matrix_arr[1]),
          Graph(masked_data, similarity_matrix_arr[2]),
          Graph(masked_data, similarity_matrix_arr[3])]


# graphs = [Graph(masked_data, similarity_matrix_arr[0])]

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
        optimizer.step()  # 更新参数
        if epoch % 10 == 0:
            print('Epoch: {}, Training Loss {:.4f}'.format(epoch, loss.item()))

    return model


views = []
n_epochs = 1000
# 训练
for i in range(len(graphs)):
    model = scGNN(graphs[i])
    optimizer = torch.optim.Adam(model.parameters())
    model = train_scGNN_wrapper(model, n_epochs, graphs[i], optimizer)
    model = model.to(device)
    # 利用未mask的矩阵，构造图，丢入训练好的model，得到中间层embedding
    embedding = model.get_embedding(Graph(scDataNorm, similarity_matrix_arr[i]))
    views.append(embedding.detach().cpu().numpy())

'''
    利用CPM-Net里面介绍的类似子空间的方式去融合Embedding（实际就是对一个Cell的不同的set（view）的融合）        
'''

# view的个数
view_num = len(views)

# 每个view的特征长度可能不一样 (在这里是一样的)
view_feat = []
for i in range(view_num):
    view_feat.append(views[i].shape[1])
sample_num = views[0].shape[0]

# 接下来对现有的数据做一个train和test的划分
train_len = int(sample_num * 1)
test_len = sample_num - train_len

'''
    查看所有view的类别分布情况    
'''
for i in range(view_num):
    tsne = TSNE()
    test_h_2d = tsne.fit_transform(views[i])
    plt.scatter(test_h_2d[:, 0], test_h_2d[:, 1], c=scLabels)
    plt.title('view'+str(i))
    plt.show()

# 把所有的view连接在一起
data_embeddings = np.concatenate(views, axis=1).astype(np.float64)
# 做一个z-score归一化
data_embeddings = z_score_Normalization(data_embeddings)
data_embeddings = torch.from_numpy(data_embeddings).float()
labels_tensor = torch.from_numpy(scLabels).view(1, scLabels.shape[0]).long()

# 这里查看下 Data Embedding的情况
tsne = TSNE()
test_h_2d = tsne.fit_transform(data_embeddings)
plt.scatter(test_h_2d[:, 0], test_h_2d[:, 1], c=scLabels)
plt.title('Data embeddings')
plt.show()

# 在这里做一个随机打乱的操作
idx = np.array([i for i in range(len(data_embeddings))])
np.random.shuffle(idx)

train_data = data_embeddings[idx[:train_len], :]
test_data = data_embeddings[idx[train_len:], :]

train_labels = labels_tensor[:, idx[:train_len]]
test_labels = labels_tensor[:, idx[train_len:]]

# 对train data做一个可视化
tsne = TSNE()
train_h_2d = tsne.fit_transform(train_data)
plt.scatter(test_h_2d[:,0], test_h_2d[:, 1],c=train_labels)
plt.title('Train_data 分布')
plt.show()


# lsd_dim 作为超参数可调
model = CPMNets(view_num, train_len, test_len, view_feat, lsd_dim=256)


n_epochs = 5000
# n_epochs = 100

# 开始训练
model.train_model(train_data, train_labels, n_epochs, lr=[0.0003, 0.0003])

# 对test_h进行adjust（按照论文的想法，保证consistency）
# model.test(test_data, n_epochs)

train_h = model.get_h_train()
# test_h = model.get_h_test()


# test_h = test_H.detach().numpy()
# test_h_path = os.path.join(os.getcwd(), "test_h.npy")
# np.save(test_h_path, test_H)
# 后面拿到test_h之后做一个k-means聚类：待解决

'''
    test_h做一个 k-means聚类
'''
# test_h = np.load(os.path.join(os.getcwd(),'test_h.npy'))
train_h = train_h.detach().cpu().numpy()
model = cluster.KMeans(n_clusters=6, max_iter=500, init="k-means++")
model.fit(train_h)

# 数据可视化
# 利用t-sne降维
tsne = TSNE()
train_h_2d = tsne.fit_transform(train_h)
plt.scatter(train_h_2d[:,0], train_h_2d[:, 1],c=model.labels_)
plt.title('Train_h_kmeans')
plt.show()

# 最后进行一个分类
# label_pre = torch.from_numpy(Classify(train_H, test_H, train_labels)).view(1, -1).long()
#
# print("Prediction Accuracy: %.3f" % ((label_pre == test_labels).sum().flaot()/(test_len)))