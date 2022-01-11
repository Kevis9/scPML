import numpy as np
import copy
from sklearn.neighbors import kneighbors_graph
import torch
from torch_geometric.data import Data as geoData
import networkx as nx
import csv


def Normalization(data):
    '''
    归一化处理
    :param data:
    :return:
    '''
    row_sum = np.sum(data, axis=1)
    mean_transcript = np.mean(row_sum)
    row_sum[np.where(row_sum==0)] = 1   #对0的部分做一个处理，防止除数为0的异常
    data_norm = (data / row_sum.reshape(-1, 1)) * mean_transcript
    return data_norm


def Mask(data, masked_prob):
    '''
        主动mask处理
    '''
    index_pair = np.where(data != 0)
    seed = 1
    np.random.seed(seed)
    masking_idx = np.random.choice(index_pair[0].shape[0], int(index_pair[0].shape[0] * masked_prob), replace=False)
    # to retrieve the position of the masked: data_train[index_pair_train[0][masking_idx], index_pair[1][masking_idx]]
    X = copy.deepcopy(data)
    X[index_pair[0][masking_idx], index_pair[1][masking_idx]] = 0

    return X, index_pair, masking_idx


def Cell_graph(data):
    '''
    :param data: 表达矩阵
    :return: 返回Cell similarity的图结构
    '''
    k = 2
    A = kneighbors_graph(data, k, mode='connectivity', include_self=False)
    G = nx.from_numpy_matrix(A.todense())
    edges = []
    # 这里认为是无向图，强行变成对称矩阵，多出来的边不用管
    for (u, v) in G.edges():
        edges.append([u, v])
        edges.append([v, u])
    edges = np.array(edges).T
    edges = torch.tensor(edges, dtype=torch.long)
    feat = torch.tensor(data, dtype=torch.float)
    # 将节点信息和边的信息放入特定类中
    G_data = geoData(x=feat, edge_index=edges)
    return G_data


def readData(dataPath, labelPath):
    '''
    读取数据
    :param Single cell的表达矩阵Path
    :param 标签Path
    :return: 返回Numpy类型数组（表达矩阵，标签，基因名）
    '''
    print('Reading data...')
    with open(dataPath) as fp:
        data = list(csv.reader(fp))
        # 一行行读取
        gene_names = data[0]
        del(gene_names[0]) # 第一个元素是空字符串
        # 为了方便处理，全部转成小写
        for i in range(len(gene_names)):
            gene_names[i] = gene_names[i].lower()
        print('Gene names:', gene_names)

        data = np.array(data[1:])[:, 1:].astype(np.float64)
        fp.close()

    with open(labelPath) as fp:
        labels = list(csv.reader(fp))[1:]
        classes = dict()
        cnt = 1
        for i in range(len(labels)):
            if labels[i][0] not in classes.keys():
                classes[labels[i][0]] = cnt
                cnt += 1

        # 对label进行一个标志
        for i in range(len(labels)):
            labels[i][0] = classes[labels[i][0]]

        labels = np.array(labels).astype(np.int64).reshape(-1)
        print("Cell type: ")
        for i in classes.keys():
            print("{} : {}".format(i, classes[i]), end=' ')
        print('')
        fp.close()

    print('Data shape is :{}'.format(data.shape))  # (samples,genes)
    return data.astype(np.float64), labels.astype(np.int64), gene_names

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
    # print(len(gene_set))
    return gene_set


import matplotlib.pyplot as plt

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



