'''
    Version 2:
    Cell-type classification:
        1. Improving Single-Cell RNA-seq Clustering by Integrating Pathways:
            1.1 利用Pathway数据将表达矩阵分成很4个不同的Similarity矩阵
                原论文提供了R源码，利用AUCell计算Pathway的score，然后将AUCell score和Single cell data用
                SNF方法融合，得到Similarity matrix(samples * samples)

        2. SCGNN: scRNA-seq Dropout Imputation via Induced Hierarchical Cell Similarity Graph
            2.1 对表达矩阵进行masked
            2.2 构造GeoData时候，Similarity矩阵使用(1)中给出的矩阵和之前不同的是，不是用masked数据进行Simiarity matrix的构造)
                但每一个node上的feature，都是masked之后的数据
            2.3 将GeoData丢入到GCN中进行训练

            最后得到4个view 

        3. CPM-Nets: Cross Partial Multi-View Networks
            利用这篇提出的CPM-Nets方法对4个view进行融合，得到每个cell的representation
'''

import torch
from torch import nn
from utils import Normalization, Mask, Cell_graph, readData, setByPathway
from Model import scGNN, CPMNets
from torch.utils.data import Dataset
import numpy as np

# 数据读取
# data: 表达矩阵
# labels: 细胞类别标签
data, labels, gene_names = readData('./Data1.csv', './label1.csv')

# 对基因集进行划分
gene_set = setByPathway(data, labels, gene_names,'./pathway/biase_150_mouse.csv')

print("Gene set's length(view_num) is %d"%(len(gene_set)))

# 对每个set做一个标准化
for i in range(len(gene_set)):
    gene_set[i] = Normalization(gene_set[i])

#开始训练：训练每个set
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = []     #每一个set的模型
embeddings = [] #每一个set的embedding (view)
i = 0
for i in range(len(gene_set)):
    data = gene_set[i] # n*feat 细胞数*基因数
    print("View {}:".format(i))
    n_epoch = 40
    masked_prob = min(len(data.nonzero()[0]) / (data.shape[0] * data.shape[1]), 0.3)
    masked_data, index_pair, masking_idx = Mask(data, masked_prob)
    G_data = Cell_graph(masked_data)
    model = scGNN(G_data=G_data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=5e-4)
    for epoch in range(n_epoch):
        '''
        把所有非零的元素比例找算出来，0.3作为上界
        '''
        model.train()
        optimizer.zero_grad() # 清除梯度
        # 先拿到train_data, 然后找到train_data里面的mask位置
        pred = model(G_data)
        # 这里的语法我们可以这样来看: pred是一个二维数组, 先 pred[:]获得所有行(省去了列，默认是获取所有列)
        # 然后得到的结果再进行索引 [index,index] 对应 [行，列]
        # dropout_pred = pred[:][
        #     index_pair[0][masking_idx], index_pair[1][masking_idx]]
        dropout_pred = pred[
            index_pair[0][masking_idx], index_pair[1][masking_idx]]
        dropout_true = data[index_pair[0][masking_idx], index_pair[1][masking_idx]]
        loss_fct = nn.MSELoss()
        loss = loss_fct(dropout_pred.view(1,-1), torch.tensor(dropout_true, dtype=torch.float).to(device).view(1,-1))

        loss.backward()
        optimizer.step()  # 更新参数
        if epoch % 10 == 0:
            print('Epoch: {}, Training Loss {:.4f}'.format(epoch, loss.item()))

    G_data = Cell_graph(data)
    models.append(model)
    embedding = model.get_embedding(G_data).detach().numpy()
    embeddings.append(embedding)
    print('embedding\'s shape is : {} (样本数, 特征数)'.format(embedding.shape))


'''
    利用CPM-Net里面介绍的类似子空间的方式去融合Embedding（实际就是对一个Cell的不同的set（view）的融合）        
'''
# view的个数
view_num = len(embeddings)

#每个view的特征长度可能不一样
view_feat = []
for i in range(view_num):
    view_feat.append(embeddings[i].shape[1])

sample_num = embeddings[0].shape[0]

# 接下来对现有的数据做一个train和test的划分
spilit = [0.8, 0.2]
train_len = int(sample_num * 0.8)
test_len = sample_num - train_len

# 数据准备
for i in range(len(embeddings)):
    embeddings[i] = np.array(embeddings[i])
data_embeddings = np.concatenate(embeddings, axis=1).astype(np.float64) # 把所有的view连接在一起
data_embeddings = torch.from_numpy(data_embeddings).float() #转为Tensor
labels_tensor = torch.from_numpy(labels).view(1, labels.shape[0]).long()

train_data = data_embeddings[:train_len,:]
test_data = data_embeddings[train_len:, :]

train_labels = labels_tensor[:,:train_len]
test_labels = labels_tensor[train_len:,:]


# lsd_dim 作为超参数可调
model = CPMNets(view_num, train_len, test_len, view_feat, lsd_dim=2)

# train H 部分
n_epoch = 25000

# 开始训练
model.train_model(train_data, train_labels, n_epoch, lr=[0.001, 0.001])