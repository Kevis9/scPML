import os.path
import numpy as np
import copy
from sklearn.neighbors import kneighbors_graph
import torch
from torch_geometric.data import Data as geoData
import networkx as nx
import csv
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import umap
import seaborn as sns

result_path = os.path.join(os.getcwd(),'result')
# 这里给出human-mouse(transfer across species)里面的类别
# label_name = ['acinar', 'activated_stellate', 'alpha', 'beta', 'delta', 'ductal', 'endothelial', 'gamma', 'macrophage', 'mast', 'quiescent_stellate']

def Normalization(data):
    '''
    scGCN中的归一化处理，对表达矩阵的每一个表达量做一个平均化
    :param data: 矩阵 (cells * genes)
    :return: 返回归一化的矩阵
    '''
    row_sum = np.sum(data, axis=1)
    mean_transcript = np.mean(row_sum)
    data_norm = (data / row_sum.reshape(-1, 1)) * mean_transcript
    return data_norm


def z_score_Normalization(data):
    '''
    利用z-score方法做一个batch normalization
    :param data: 矩阵，（样本 * 特征）, 二维数组
    :return:
    '''
    means = np.mean(data, axis=0)
    standard = np.std(data, axis=0)
    print(np.where(standard==0))

    return (data - means)/standard


def Mask_Data(data, masked_prob):
    '''
    :param data: 表达矩阵 (cells * genes)
    :param masked_prob: mask概率
    :return:
        1. X: mask的表达矩阵，
        2. index_pair: 矩阵中不为0的索引，[(行),(列)],
        3. masking_idx: 随机选择index_pair中行列的下标
        X[index_pair[0][masking_idx], index_pair[1][masking_idx]] 就是指被masked的数据
    '''
    index_pair = np.where(data != 0)
    seed = 1
    np.random.seed(seed)
    masking_idx = np.random.choice(index_pair[0].shape[0], int(index_pair[0].shape[0] * masked_prob), replace=False)
    # to retrieve the position of the masked: data_train[index_pair_train[0][masking_idx], index_pair[1][masking_idx]]
    X = copy.deepcopy(data)
    X[index_pair[0][masking_idx], index_pair[1][masking_idx]] = 0

    return X, index_pair, masking_idx


def Graph(data, mat_similarity, k):
    '''
    :param data: 表达矩阵 (被mask的矩阵)
    :param mat_similarity: 邻接矩阵 (ndarray)
    :return: 返回Cell similarity的图结构
    '''
    # k = 4

    # 要对mat_similarity取前K个最大的weight作为neighbors
    k_idxs = []
    # 现将对角线部分全部设为0, 避免自己做自己的邻居
    mat_similarity[np.diag_indices_from(mat_similarity)] = 0

    for i in range(mat_similarity.shape[0]):
        top_k_idx = mat_similarity[i].argsort()[::-1][0:k]
        k_idxs.append(top_k_idx)

    mat_similarity = np.zeros(shape=mat_similarity.shape)
    for i in range(mat_similarity.shape[0]):
        mat_similarity[i, k_idxs[i]] = 1
    mat_similarity = mat_similarity.astype(np.int64)
    G = nx.from_numpy_matrix(np.matrix(mat_similarity))


    # np.savetxt('./Pathway_similarity.csv', mat_similarity,delimiter=',', fmt='%i')
    # 这里稍微修改下，尝试用原来Huang的Similarity matrix来做
    # A = kneighbors_graph(data, k, mode='connectivity', include_self=False) # 拿到Similarity矩阵
    # np.savetxt('./Huang_Similarity.csv', A.todense(), delimiter=',', fmt='%i')
    # print(A.todense())
    # G = nx.from_numpy_matrix(A.todense())

    edges = []
    # 把有向图转为无向图
    for (u, v) in G.edges():
        edges.append([u, v])
        edges.append([v, u])
    edges = np.array(edges).T
    edges = torch.tensor(edges, dtype=torch.long)
    feat = torch.tensor(data, dtype=torch.float)
    # 将节点信息和边的信息放入特定类中
    G_data = geoData(x=feat, edge_index=edges)
    return G_data


def readSCData(dataPath, labelPath):
    '''
    读取数据, 数据格式需要满足一定格式
    表达矩阵第一列是cell id, 第一行是名称
    label矩阵只有一列

    :param Single cell的表达矩阵Path
    :param 标签Path
    :return: 返回Numpy类型数组（表达矩阵，标签）
    '''
    print('Reading data...')
    with open(dataPath) as fp:
        data = list(csv.reader(fp))
        data = np.array(data[1:])[:, 1:].astype(np.float64)
        fp.close()

    with open(labelPath) as fp:
        labels = list(csv.reader(fp))[1:]

        labels = (np.array(labels)[:,:]).astype(np.int64).reshape(-1)
        fp.close()

    print('表达矩阵的shape为 :{}'.format(data.shape))  # (samples,genes)
    return data.astype(np.float64), labels.astype(np.int64)


def setByPathway(data, labels, gene_names, path):
    '''
    :param data: 训练数据
    :param labels: 标签
    :param gene_names: 基因的名称
    :param path: pathway文件所在路径
    :return:
    '''
    with open(path) as fp:
        pathway = list(csv.reader(fp))
        groups = dict()
        # 把每一个基因的组建立一个映射
        for i in range(len(pathway)):
            for j in range(1, len(pathway[i])):
                # 注意全部转为小写
                groups[pathway[i][j].lower()] = int(pathway[i][0])
        fp.close()
    # 记录每个组（比如第一组）基因在data中的下标位置, 下标代表的是某一组
    # +1的原因是组从1开始计算
    gene_idx = [[] for g in range(len(pathway)+1)]
    for i in range(len(gene_names)):
        if gene_names[i] in groups.keys():
            # 如果不在Gene不再Pathway数据集里面，就不要了
            gene_idx[groups[gene_names[i]]].append(i) # i是矩阵里面gene的下标，从0开始

    gene_set = []
    for group_id in range(len(pathway)):
        # 控制每一组的基因数目都要大于2
        if len(gene_idx[group_id+1]) < 3:
            continue
        gene_set.append(data[:,gene_idx[group_id+1]]) #拿第1、2、3...组基因放到gene_set里面

    return gene_set


def readSimilarityMatrix(path):
    with open(path) as fp:
        mat_similarity = np.array(list(csv.reader(fp))[1:])[:,1:]
        mat_similarity = mat_similarity.astype(np.float64) # 记得做个类型转换
        fp.close()
    return mat_similarity




def lossPolt(r_loss, c_loss, n_epoch):
    fig, ax = plt.subplots()
    x = [i for i in range(n_epoch)]
    ax.plot(x, r_loss, label='Reconstruction')
    ax.plot(x, c_loss, label='Classification')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss')
    ax.legend()
    plt.show()


def Classify(lsd1, lsd2, label):
    """In most cases, this method is used to predict the highest accuracy.
    :param lsd1: train set's latent space data
    :param lsd2: test set's latent space data
    :param label: label of train set
    :return: Predicted label
    """
    F_h_h = np.dot(lsd2, np.transpose(lsd1))
    label = label.reshape(len(label), 1) - 1
    enc = OneHotEncoder()
    a = enc.fit_transform(label)
    label_onehot = a.toarray()
    label_num = np.sum(label_onehot, axis=0)
    F_h_h_sum = np.dot(F_h_h, label_onehot)
    F_h_h_mean = F_h_h_sum / label_num
    label_pre = np.argmax(F_h_h_mean, axis=1) + 1
    return label_pre


def sharedGeneMatrix(path1, path2):

    pd1 = pd.read_csv(path1)
    # 这地方给的pd2要转置
    pd2 = pd.read_csv(path2, index_col=0).T
    pd3 = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/mouse_VISp_gene_expression_matrices_2018-06-14/mouse_VISp_2018-06-14_genes-rows.csv')
    pd2 = pd.DataFrame(data=pd2.values, columns=pd3['gene_symbol'].tolist())
    print("数据读取完成")

    names1 = pd1.columns
    names2 = pd2.columns
    print(names1)
    print(names2)

    # 取交集
    shared_names = names1.intersection(names2)
    print(shared_names)

    # 拿到human和mouse的DataFrame
    matrix1 = pd1[shared_names]
    matrix2 = pd2[shared_names]

    # 保存数据
    matrix1.to_csv('mouse1_pancreas.csv')
    matrix2.to_csv('mouse_VISP.csv')



def showClusters(data, label, title):
    '''
    可视化聚类的函数
    :param data: 表达矩阵
    :param label: 样本的标签
    :param title: 可视化窗口的titleplt.scatter
    '''
    # 这里尝试用UAMP进行降维处理
    # To ensure that results can be reproduced exactly UMAP allows the user to set a random seed state
    umap_model = umap.UMAP(random_state=29)
    data_2d = umap_model.fit_transform(data)
    # tsne = TSNE()
    # data_2d = tsne.fit_transform(data)

    data = {
        'x':data_2d[:,0],
        'y':data_2d[:,1],
        'label':label
    }
    # print(title)
    # print(set(label))
    df = pd.DataFrame(data=data)
    # arr = [(i+1) for i in range(11)] # 1...11

    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df, x='x', y='y', hue='label', palette='deep', s=6)
    plt.legend(loc=3, bbox_to_anchor=(1, 0)) # 设置图例位置
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.title(title)
    plt.savefig(os.path.join(result_path,title+'.png'))
    # plt.show()

    # 用matplot绘制图片，想要画出好一点的图片不方便
    # plt.scatter(data_2d[:, 0], data_2d[:, 1], c=label, cmap='Spectral', s=5)
    # # plt.gca().set_aspect('equal', 'datalim')
    #
    # classes_num = len(set(label))
    # plt.colorbar().set_ticks(np.arange(1,classes_num+1))
    #
    # plt.title(title)
    # plt.show()



