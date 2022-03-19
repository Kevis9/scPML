import os.path
import numpy as np
import copy
# from sklearn.neighbors import kneighbors_graph
import torch
from torch_geometric.data import Data as geoData
import networkx as nx
import csv
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.manifold import TSNE
import umap
import seaborn as sns

RESULT_PATH = os.path.join(os.getcwd(), 'result')

def sc_normalization(data):
    '''
    scGCN中的归一化处理，对表达矩阵的每一个表达量做一个平均加权
    :param data: 矩阵 (cells * genes)
    :return: 返回归一化的矩阵
    '''
    row_sum = np.sum(data, axis=1)
    mean_transcript = np.mean(row_sum)
    data_norm = (data / row_sum.reshape(-1, 1)) * mean_transcript
    return data_norm


def z_score_normalization(data):
    '''
    利用z-score方法做一个batch normalization
    :param data: 矩阵，（样本 * 特征）, 二维数组
    :return:
    '''
    means = np.mean(data, axis=0)
    standard = np.std(data, axis=0)
    print(np.where(standard==0))

    return (data - means)/standard


def mask_data(data, masked_prob):
    '''
    :param data: 表达矩阵 (cells * genes)
    :param masked_prob: mask概率
    :return:
        1. X: mask后的表达矩阵，
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


def construct_graph(data, similarity_mat, k):
    '''
    :param data: 表达矩阵 (被mask的矩阵)
    :param similarity_mat: 邻接矩阵 (ndarray)
    :return: 返回Cell similarity的图结构
    '''

    # 要对similarity_mat取前K个最大的weight作为neighbors
    k_idxs = []
    # 现将对角线部分全部设为0, 避免自己做自己的邻居
    similarity_mat[np.diag_indices_from(similarity_mat)] = 0

    for i in range(similarity_mat.shape[0]):
        top_k_idx = similarity_mat[i].argsort()[::-1][0:k]
        k_idxs.append(top_k_idx)

    similarity_mat = np.zeros(shape=similarity_mat.shape)
    similarity_mat[:, k_idxs[i]] = 1
    # for i in range(similarity_mat.shape[0]):
    #     similarity_mat[i, k_idxs[i]] = 1
    similarity_mat = similarity_mat.astype(np.int64)
    graph = nx.from_numpy_matrix(np.matrix(similarity_mat))

    # np.savetxt('./Pathway_similarity.csv', mat_similarity,delimiter=',', fmt='%i')
    # 这里稍微修改下，尝试用原来Huang的Similarity matrix来做
    # A = kneighbors_graph(data, k, mode='connectivity', include_self=False) # 拿到Similarity矩阵
    # np.savetxt('./Huang_Similarity.csv', A.todense(), delimiter=',', fmt='%i')
    # print(A.todense())
    # G = nx.from_numpy_matrix(A.todense())

    edges = []
    # 把有向图转为无向图
    for (u, v) in graph.edges():
        edges.append([u, v])
        edges.append([v, u])
    edges = np.array(edges).T
    edges = torch.tensor(edges, dtype=torch.long)
    feat = torch.tensor(data, dtype=torch.float)
    # 将节点信息和边的信息放入特定类中
    g_data = geoData(x=feat, edge_index=edges)
    return g_data


def read_data_label(data_path, label_path):
    '''
    读取数据, 数据格式需要满足一定格式
    表达矩阵第一列是cell id, 第一行是名称
    label矩阵只有一列

    :param Single cell的表达矩阵Path
    :param 标签Path
    :return: 返回Numpy类型数组（表达矩阵，标签）
    '''
    print('Reading data...')
    data_df = pd.read_csv(data_path, index_col=0)
    data = data_df.to_numpy()

    label_df = pd.read_csv(label_path)
    label = label_df.to_numpy().reshape(-1)
    # with open(data_path) as fp:
    #     data = list(csv.reader(fp))
    #     data = np.array(data[1:])[:, 1:].astype(np.float64)
    #     fp.close()

    # with open(label_path) as fp:
    #     labels = list(csv.reader(fp))[1:]
    #
    #     labels = (np.array(labels)[:,:]).astype(np.int64).reshape(-1)
    #     fp.close()

    print('表达矩阵的shape为 :{}'.format(data.shape))  # (samples,genes)
    print('label的shape为 : {}'.format(label.shape))
    return data.astype(np.float64), label.astype(np.int64)




def read_similarity_mat(path):

    mat_df = pd.read_csv(path, index_col=0)
    similarity_mat = mat_df.to_numpy()
    # with open(path) as fp:
    #     mat_similarity = np.array(list(csv.reader(fp))[1:])[:,1:]
    #     mat_similarity = mat_similarity.astype(np.float64) # 记得做个类型转换
    #     fp.close()
    return similarity_mat.astype(np.float64)




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
    # print(a)
    label_onehot = a.toarray()
    label_num = np.sum(label_onehot, axis=0)
    F_h_h_sum = np.dot(F_h_h, label_onehot)
    F_h_h_mean = F_h_h_sum / label_num
    label_pre = np.argmax(F_h_h_mean, axis=1) + 1
    return label_pre


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
    # tsne = TSNE() # TSNE进行降维处理
    # data_2d = tsne.fit_transform(data)

    data = {
        'x':data_2d[:,0],
        'y':data_2d[:,1],
        'label':label
    }

    df = pd.DataFrame(data=data)
    # plt.figure(figsize=(8,5))
    sns.scatterplot(data=df, x='x', y='y', hue='label', palette='deep', s=8)
    plt.legend(loc=3, bbox_to_anchor=(1, 0)) # 设置图例位置
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.title(title)
    plt.savefig(os.path.join(RESULT_PATH, title + '.png'), dpi=600, format='svg')
    # plt.show()



