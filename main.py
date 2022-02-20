'''
    Version 2:
    Cell-type classification:
        1. Improving Single-Cell RNA-seq Clustering by Integrating Pathways:
            1.1 利用Pathway数据将表达矩阵分成很4个不同的Similarity矩阵
                原论文提供了R源码，利用AUCell计算Pathway的score，然后将AUCell score和Single cell data用
                SNF方法融合，得到Similarity matrix(cells * cells)

        2. SCGNN: scRNA-seq Dropout Imputation via Induced Hierarchical Cell Similarity Graph
            2.1 对表达矩阵进行masked
            2.2 构造GeoData时候，Similarity矩阵使用(1)中给出的矩阵和之前不同的是，不是用masked数据进行Simiarity matrix的构造)
                但每一个node上的feature，都是masked之后的数据
            2.3 将GeoData丢入到GCN中进行训练

            最后得到4个view 

        3. CPM-Nets: Cross Partial Multi-View Networks
            利用这篇提出的CPM-Nets方法对4个view进行融合，得到每个cell的representation
'''
import os.path

import torch
from torch import nn
from utils import Normalization, Mask_Data, Graph, readSCData, setByPathway, readSimilarityMatrix, Classify
from Model import scGNN, CPMNets
from torch.utils.data import Dataset
import numpy as np



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# 数据读取, 得到单细胞表达矩阵和标签

scData, scLabels = readSCData(os.path.join(os.getcwd(), "Single_Cell_Sequence", "mat_gene.csv"), os.path.join(os.getcwd(), "Single_Cell_Sequence", "label.csv"))

# 对单细胞表达矩阵做归一化
scDataNorm = Normalization(scData)

'''
    对数据进行随机mask (仅仅模拟Dropout event)
'''
# 概率
masked_prob = min(len(scDataNorm.nonzero()[0]) / (scDataNorm.shape[0] * scDataNorm.shape[1]), 0.3)

# 得到被masked之后的数据
masked_data, index_pair, masking_idx = Mask_Data(scDataNorm, masked_prob)

'''
    根据Cell Similarity矩阵，构造出Graph来，每个节点的feature是被masked之后的矩阵        
'''
base_path = os.path.join(os.getcwd(), "Similarity_Matrix")
similarity_matrix_arr = [readSimilarityMatrix(os.path.join(base_path, 'KEGG_yan_human.csv')),
                         readSimilarityMatrix(os.path.join(base_path, 'Reactome_yan_human.csv')),
                         readSimilarityMatrix(os.path.join(base_path, 'Wikipathways_yan_human.csv')),
                         readSimilarityMatrix(os.path.join(base_path, 'yan_yan_human.csv'))]

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
        optimizer.step()  # 更新参数
        if epoch % 10 == 0:
            print('Epoch: {}, Training Loss {:.4f}'.format(epoch, loss.item()))

    return model


views = []
n_epochs = 60
# 训练
for i in range(len(graphs)):
    model = scGNN(graphs[i])
    optimizer = torch.optim.Adam(model.parameters())
    model = train_scGNN_wrapper(model, n_epochs, graphs[i], optimizer)
    model = model.to(device)
    # 利用未mask的矩阵，构造图，丢入训练好的model，得到中间层embedding
    embedding = model.get_embedding(Graph(scDataNorm, similarity_matrix_arr[i]))
    views.append(embedding.detach().numpy())


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
train_len = int(sample_num * 0.7)
test_len = sample_num - train_len

# 把所有的view连接在一起
data_embeddings = np.concatenate(views, axis=1).astype(np.float64)
data_embeddings = torch.from_numpy(data_embeddings).float()
labels_tensor = torch.from_numpy(scLabels).view(1, scLabels.shape[0]).long()

train_data = data_embeddings[:train_len, :]
test_data = data_embeddings[train_len:, :]

train_labels = labels_tensor[:, :train_len]
test_labels = labels_tensor[train_len:, :]

# lsd_dim 作为超参数可调
model = CPMNets(view_num, train_len, test_len, view_feat, lsd_dim=256)


# n_epochs = 15000
n_epochs = 100

# 开始训练
model.train_model(train_data, train_labels, n_epochs, lr=[0.0003, 0.0003])

# 对test_h进行adjust（按照论文的想法，保证consistency）
model.test(test_data, n_epochs)

train_H = model.get_h_train()
test_H = model.get_h_test()

print("test h is \n",test_H)
test_H = test_H.detach().numpy()
test_h_path = os.path.join(os.getcwd(), "test_h.npy")
np.save(test_h_path, test_H)
# 后面拿到test_h之后做一个k-means聚类：待解决


# 最后进行一个分类
label_pre = torch.from_numpy(Classify(train_H, test_H, train_labels)).view(1, -1).long()

print("Prediction Accuracy: %.3f" % ((label_pre == test_labels).sum().flaot()/(test_len)))