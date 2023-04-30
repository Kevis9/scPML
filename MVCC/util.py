import math
import os.path
import random
import numpy as np
import copy
import sklearn
from sklearn.neighbors import kneighbors_graph
from torch_sparse import SparseTensor
import torch
from torch_geometric.data import Data as geoData
import networkx as nx
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import pandas as pd
import umap
from sklearn.metrics import silhouette_score, adjusted_rand_score, accuracy_score, f1_score
import seaborn as sns
import wandb
import scipy.spatial as spt
from random import sample
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
import torch.nn.functional as F
hue = []

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    # 这个很重要
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

def mean_norm(data):
    '''
    scGCN中的标准化处理，对表达矩阵的每一个表达量做一个平均加权
    :param data: 矩阵 (cells * genes)
    :return: 返回归一化的矩阵
    '''
    row_sum = np.sum(data, axis=1)
    mean_transcript = np.mean(row_sum)
    print("细胞平均表达量是 {:.3f}".format(mean_transcript))
    # 防止出现除0的问题
    row_sum[np.where(row_sum == 0)] = 1

    scale_factor = 1e4
    # data_norm = np.log1p((data / row_sum.reshape(-1 ,1))*scale_factor)
    data_norm = (data / row_sum.reshape(-1, 1)) * mean_transcript

    return data_norm


def z_score_scale(data):
    '''
    利用z-score方法做一个batch normalization
    :param data: 矩阵，（样本 * 特征）, 二维数组
    :return:
    '''
    means = np.mean(data, axis=0)
    standard = np.std(data, axis=0)

    return (data - means) / standard


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
    idx = np.random.choice(index_pair[0].shape[0], int(index_pair[0].shape[0] * masked_prob), replace=False)

    # masking_idx = [idx, idx]
    # to retrieve the position of the masked: data_train[index_pair_train[0][masking_idx], index_pair[1][masking_idx]]
    X = copy.deepcopy(data)
    mask_idx = [index_pair[0][idx], index_pair[1][idx]]
    X[mask_idx[0], mask_idx[1]] = 0

    return X, mask_idx


def mask_cells(data, masked_prob):
    '''
        对data中每个cell进行随机的mask
    '''
    seed = 1
    random.seed(seed)
    mask_rows = []
    mask_cols = []
    num = int(data.shape[1] * masked_prob)
    mask_col_idx = random.sample(range(data.shape[1]), num)
    for i in range(data.shape[0]):
        mask_rows += [i] * num
        mask_cols += mask_col_idx
        # mask_col_idx = random.sample(range(data.shape[1]), num)
        # mask_cols += mask_col_idx
    X = copy.deepcopy(data)
    # print(index_pair[0])
    X[mask_rows, mask_cols] = 0
    mask_idx = [mask_rows, mask_cols]
    return X, mask_idx


def mask_column(data, masked_prob, cols):
    tmp_data = data[:, cols]
    index_pair = np.where(tmp_data != 0)
    seed = 1
    random.seed(seed)  # 为了复现
    idx = random.sample(range(index_pair[0].shape[0]), int(index_pair[0].shape[0] * masked_prob))  # 无重复采样
    mask_rows = index_pair[0][idx]
    # cols 代表原来数据的列下标
    cols = np.array(cols)
    # print(index_pair[1][idx])
    mask_cols = cols[index_pair[1][idx]]
    mask_idx = [mask_rows, mask_cols]
    # to retrieve the position of the masked: data_train[index_pair_train[0][masking_idx], index_pair[1][masking_idx]]
    X = copy.deepcopy(data)
    # print(index_pair[0])
    X[mask_idx[0], mask_idx[1]] = 0
    return X, mask_idx


def construct_graph_with_knn(data, k=2):
    A = kneighbors_graph(data, k, mode='connectivity', include_self=False)  # 拿到Similarity矩阵
    G = nx.from_numpy_matrix(A.todense())
    edges = []
    # 把有向图转为无向图
    for (u, v) in G.edges():
        edges.append([u, v])
        edges.append([v, u])
    edges = np.array(edges).T
    edges = torch.tensor(edges, dtype=torch.long)
    feat = torch.tensor(data, dtype=torch.float)
    # 将节点信息和边的信息放入特定类中
    g_data = geoData(x=feat, edge_index=edges)
    return g_data


def check_out_similarity_matrix(sm, labels, k, sm_name):
    '''
        检查一下相似矩阵中类型相同的细胞是否存在边
        给出具体的信息
    '''
    sm = construct_adjacent_matrix_with_MNN(sm, k)
    types = sorted(list(set(labels)))

    '''
        这里查看CD4 + T cell的边的情况
    '''
    # cd4_idx = np.where(labels=='CD4+ T cell')[0]
    # total_edges = sm[cd4_idx, :].sum()
    # print("CD4+ T cell总共有 {:} 边".format(total_edges//2))
    # for item in types:
    #     idx = np.where(labels == item)[0]
        # print(sm[cd4_idx, idx].sum())
        # print("和 {:} 有 {:}边".format(item, sm[cd4_idx][:, idx].sum()//2))

    '''
        这里换一种评价的方式：
            delta = max(0, (E1- E2) / n(n-1))
            E1:代表内部边数目
            E2:代表和其他类别相连的总边数
            n代表节点数目
    '''
    # for cell_type in types:
    #     if not cell_type == 'schwann':
    #         continue
    #     idx = np.where(labels == cell_type)[0]
    #     # 这个type每个节点对其他节点边的和
    #     row_sum = sm[idx, :].sum(axis=0)
    #     # 这个type节点之间的边数
    #     E1 = row_sum[idx].sum()
    #     E2 = row_sum.sum() - E1
    #     n = len(idx)
    #     factor = 0.3
    #     delta = max(0, (E1- factor * E2)/(n*(n-1)))
    #     print("{:}, 富集系数为{:.3f}".format(cell_type, delta))
    #
    # return None
    confusion_matrix = []
    for i in range(len(set(types))):
        confusion_matrix.append([0 for j in range(len(types))])

    for i, label in enumerate(types):
        idx = np.where(labels == label)
        label_sm = sm[idx[0], :]
        sm_sum = label_sm.sum(axis=0)
        # print("For {:}({:}), his neighbor situation is".format(label, len(idx[0])))
        for j, type_x in enumerate(types):
            type_x_idx = np.where(labels == type_x)
            # print("{:}: {:} egdes".format(type_x, sum(sm_sum[type_x_idx])))
            confusion_matrix[i][j] = sum(sm_sum[type_x_idx])

    # 检查下CD4+ 和 CD8+ 的关系(seq-well - 10x_v3)
    # print((confusion_matrix[2][2] + confusion_matrix[3][3] )/ (confusion_matrix[2][3] +confusion_matrix[3][2]) )
    # 绘制类型之间的相似矩阵
    confusion_mat = np.array(confusion_matrix)
    # 归一化
    confusion_mat = confusion_mat / np.sum(confusion_mat, axis=1).reshape(-1, 1)
    data_df = pd.DataFrame(
        confusion_mat
    )
    data_df.columns = types
    data_df.index = types

    sns.heatmap(data=data_df, cmap="Blues", cbar=True, xticklabels=True, yticklabels=True)
    plt.savefig(sm_name, dpi=300, bbox_inches="tight")
    plt.clf()





def get_similarity_matrix(data, k=2):
    '''
        利用KNN得到一个邻接矩阵
    '''
    A = kneighbors_graph(data, k, mode='connectivity', include_self=False)  # 拿到Similarity矩阵
    return A.todense()

def construct_adjacent_matrix(similarity_mat, k):
    '''
        从权重的相似矩阵得到邻接矩阵
    '''
    if k==0:
        '''
            假设k=0的时候，自己做自己的邻居，GCN退化成一个全连接
        '''
        similarity_mat = np.zeros(shape=similarity_mat.shape)
        similarity_mat[np.diag_indices_from(similarity_mat)] = 1
        return similarity_mat

    # 要对similarity_mat取前K个最大的weight作为neighbors
    k_idxs = []
    # 将对角线部分全部设为0, 避免自己做自己的邻居
    similarity_mat[np.diag_indices_from(similarity_mat)] = 0
    for i in range(similarity_mat.shape[0]):
        top_k_idx = similarity_mat[i].argsort()[::-1][0:k]
        k_idxs.append(top_k_idx)

    similarity_mat = np.zeros(shape=similarity_mat.shape)
    # 原来这一步真的很离谱，这里构造图的时候一直都错了，下面的for循环才是对的
    for i in range(similarity_mat.shape[0]):
        similarity_mat[i, k_idxs[i]] = 1

    adjacent_mat = similarity_mat.astype(np.int64)
    return adjacent_mat

def construct_adjacent_matrix_with_MNN(similarity_mat, k):
    '''
        利用MNN来构图，相比之前，MNN可以有效减少不必要的边
    '''
    # 要对similarity_mat取前K个最大的weight作为neighbors
    k_idxs = []
    # 将对角线部分全部设为0, 避免自己做自己的邻居
    similarity_mat[np.diag_indices_from(similarity_mat)] = 0
    for i in range(similarity_mat.shape[0]):
        top_k_idx = similarity_mat[i].argsort()[::-1][0:k]
        k_idxs.append(top_k_idx)
    adjacent_mat = np.zeros(shape=similarity_mat.shape)
    for i in range(similarity_mat.shape[0]):
        adjacent_mat[i, k_idxs[i]] = 1

    # 利用MNN的思想，两个节点之间如果有边的话，那么两个节点的邻居都要包含彼此
    adjacent_mat = np.logical_and(adjacent_mat, adjacent_mat.T)
    adjacent_mat = adjacent_mat.astype(np.int64)
    # print(np.sum(adjacent_mat, axis=1))
    # print(max(np.sum(adjacent_mat, axis=1)))
    return adjacent_mat

def construct_graph(data, sm_mat, k):
    '''
    :param data: 表达矩阵 (被mask的矩阵)
    :param sm_mat: 邻接矩阵 (ndarray)
    :return: 返回Cell similarity的图结构
    '''

    sm_mat = construct_adjacent_matrix_with_MNN(sm_mat, k)
    # sm_mat = construct_adjacent_matrix(sm_mat, k)

    graph = nx.from_numpy_matrix(np.matrix(sm_mat))

    edges = []
    # 把有向图转为无向图
    for (u, v) in graph.edges():
        edges.append([u, v])
        edges.append([v, u])

    edges = np.array(edges).T

    edges = torch.tensor(edges, dtype=torch.long)

    sm_mat = torch.tensor(sm_mat, dtype=torch.float)
    adj = SparseTensor.from_dense(sm_mat)
    feat = torch.tensor(data, dtype=torch.float)

    # 将节点信息和边的信息放入特定类中
    g_data = geoData(x=feat, edge_index=edges)


    # return g_data
    return (feat, adj.t())


def read_data_label_h5(path, key):
    print('Reading data...')
    data_path = os.path.join(path, 'data.h5')

    data_df = pd.read_hdf(data_path, key + '/data')
    data = data_df.to_numpy()

    label_df = pd.read_hdf(data_path, key + '/label')
    # print(label_df['type'].value_counts())

    print(label_df.iloc[:, 0].value_counts())

    label = label_df.to_numpy().reshape(-1)

    print('表达矩阵的shape为 :{}'.format(data.shape))  # (samples,genes)
    print('label的shape为 : {}'.format(label.shape))
    return data, label


# def read_data_label(data_path, label_path):
#     '''
#     读取数据, 数据格式需要满足一定格式
#     表达矩阵第一列是cell id, 第一行是名称
#     label矩阵只有一列
#
#
#     :param Single cell的表达矩阵Path
#     :param 标签Path
#     :return: 返回Numpy类型数组（表达矩阵，标签）
#     '''
#
#     print('Reading data...')
#     data_df = pd.read_csv(data_path, index_col=0)
#     data = data_df.to_numpy()
#
#     label_df = pd.read_csv(label_path)
#     label = label_df.to_numpy().reshape(-1)
#     print('表达矩阵的shape为 :{}'.format(data.shape))  # (samples,genes)
#     print('label的shape为 : {}'.format(label.shape))
#     return data, label


# def read_similarity_mat(path):
#     mat_df = pd.read_csv(path, index_col=0)
#     similarity_mat = mat_df.to_numpy()
#     return similarity_mat.astype(np.float64)


def read_similarity_mat_h5(path, key):
    # print("reading graph...")
    data_path = os.path.join(path, 'data.h5')
    mat_df = pd.read_hdf(data_path, key)

    # print(mat_df.isnull().any())
    #
    # print(np.isnan(mat_df).any())
    # print(np.isinf(mat_df).all())
    #
    # print(np.isfinite(mat_df).all())
    similarity_mat = mat_df.to_numpy()
    # print("Finish")
    return similarity_mat.astype(np.float64)


def sel_feature(data1, data2, label1, nf=3000):
    # 先去掉表达量为0的基因, 然后再做交集, 这里暂时不打算取HVG
    # sum1 = np.sum(data1, axis=0)
    # idx1 = set(np.where(sum1!=0)[0])
    # sum2 = np.sum(data2, axis=0)
    # idx2 = set(np.where(sum2!=0)[0])
    # idx = list(idx1 & idx2) #交集
    # data1 = data1[:, idx]
    # data2 = data2[:, idx]
    # print("After gene selction , ref data shape {:}, query data shape {:}".format(data1.shape, data2.shape))
    # return data1, data2

    sel_model = SelectKBest(k=nf)  # default score function is f_classif

    sel_model.fit(data1, label1)
    idx = sel_model.get_support(indices=True)

    # sel_model = SelectKBest(k=nf)
    # sel_model.fit(data2, label2)
    # idx2 = sel_model.get_support(indices=True)

    # idx = list(set(idx1) & set(idx2))
    data1 = data1[:, idx]
    data2 = data2[:, idx]
    print("After gene selction , ref data shape {:}, query data shape {:}".format(data1.shape, data2.shape))

    return data1, data2



def pre_process(data1, data2, label1, nf=2000):
    # 先选择合适的feature
    data1, data2 = sel_feature(data1, data2, label1, nf=nf)
    # scaler = StandardScaler()
    data1 = mean_norm(data1)
    data2 = mean_norm(data2)
    # data1 = scaler.fit_transform(data1)
    # data2 = scaler.transform(data2)
    # 再做ScaleData: z_score
    # data1, data2 = z_score_scale(data1), z_score_scale(data2)
    return data1, data2


def cpm_classify(lsd1, lsd2, label):
    """In most cases, this method is used to predict the highest accuracy.
    :param lsd1: train set's latent space data
    :param lsd2: test set's latent space data
    :param label: label of train set
    :return: Predicted label
    """
    F_h_h = np.dot(lsd2, np.transpose(lsd1))
    label = label.reshape(len(label), 1)
    enc = OneHotEncoder()
    a = enc.fit_transform(label)
    # print(a)
    label_onehot = a.toarray()
    label_num = np.sum(label_onehot, axis=0)
    F_h_h_sum = np.dot(F_h_h, label_onehot)
    F_h_h_mean = F_h_h_sum / label_num
    label_pre = np.argmax(F_h_h_mean, axis=1)
    return label_pre


def runUMAP(data):
    umap_model = umap.UMAP(random_state=0)
    data_2d = umap_model.fit_transform(data)
    return data_2d


def runPCA(data):
    pca = PCA(n_components=32)
    return pca.fit_transform(data)


def show_cluster(data, label, title, save_path):
    '''
    可视化聚类的函数
    :param data: 降维之后的数据(2d)
    :param label: 样本的标签
    :param title: 可视化窗口的titleplt.scatter
    '''
    save_path = os.path.join(save_path, 'image')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data = {
        'x': data[:, 0],
        'y': data[:, 1],
        'label': label
    }

    df = pd.DataFrame(data=data)
    # 去掉部分数据(为了更好的可视化)
    # df = df[~((df['x']>10) | (df['y']>10))]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='x', y='y', hue='label', palette='deep', s=3)
    plt.legend(loc=3, bbox_to_anchor=(1, 0))  # 设置图例位置
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.title(title)
    plt.savefig(os.path.join(save_path, "_".join(title.split()) + '.png'), bbox_inches='tight')  # dpi可以设置

    # 数据上报
    # wandb.save(os.path.join(save_path, "_".join(title.split()) + '.png'))
    # plt.show()


def concat_views(views):
    return np.concatenate(views, axis=1).astype(np.float64)


def batch_mixing_entropy(ref_data, query_data, L=100, M=300, K=500):
    '''
    :param ref_data:
    :param query_data:
    :param L: 次数
    :param M: 随机抽取的细胞数
    :param k: neibor数x
    :return: 返回一个BatchEntroy数组, 代表进行L次随机取样的结果
    '''
    data = np.concatenate([ref_data, query_data], axis=0)
    nbatchs = 2
    batch0 = np.concatenate([np.zeros(ref_data.shape[0]), np.ones(query_data.shape[0])])
    entropy = [0 for i in range(L)]
    kdtree = spt.KDTree(data)
    random.seed(0)
    data_idx = [p for p in range(data.shape[0])]
    for boot in range(L):
        rand_samples_idx = sample(data_idx, M)
        _, neighbor_idx = kdtree.query(data[rand_samples_idx, :], k=K)
        for i in range(len(rand_samples_idx)):
            for j in range(nbatchs):
                xi = max(1, (batch0[neighbor_idx[i]] == j).sum())
                entropy[boot] += xi * math.log(xi)
    entropy = [-(x / M) for x in entropy]
    return entropy


def encode_label(ref_label, query_label):
    '''
    :param ref_label: ref的label
    :param query_label:  query的label
    :return:
    '''
    enc = LabelEncoder()

    enc.fit(np.concatenate([ref_label, query_label]))

    return enc.transform(ref_label), enc.transform(query_label), enc


def precision_of_cell(cell_type, pred, trues):
    idx = np.array(np.where(trues == cell_type)).squeeze()
    acc = (pred[idx] == cell_type).sum() / len(idx)
    return acc



def confusion_plot(pred, true, save_name):
    print(accuracy_score(pred, true))

    name = list(set(true))
    name = [x.lower() for x in name]
    # if not "unassigned" in name:
    #     name.append("unassigned")
    name.sort()

    name_idx = {}
    for i in range(len(name)):
        name_idx[name[i]] = i

    confusion_mat = []
    print(set(true))
    print(set(name))
    # 行是true，只考虑true的部分
    for i in range(len(set(true))):
        confusion_mat.append([0 for j in range(len(name))])

    pred = list(pred)
    true = list(true)

    pred = [x.lower() for x in pred]
    true = [x.lower() for x in true]

    for i in range(len(true)):
        row = name_idx[true[i]]
        col = name_idx[pred[i]]
        confusion_mat[row][col] += 1

    ## 构造DataFrame
    confusion_mat = np.array(confusion_mat)
    # 归一化
    confusion_mat = confusion_mat / np.sum(confusion_mat, axis=1).reshape(-1, 1)
    data_df = pd.DataFrame(
        confusion_mat
    )
    data_df.columns = name
    true_name = list(set(true))
    true_name.sort()
    data_df.index = true_name

    # 将数据倒置过来
    data_df = data_df.reindex(index=data_df.index[::-1])

    print(data_df.index)
    print(data_df.columns)

    sns.heatmap(data=data_df, cmap="Blues", cbar=False, xticklabels=True, yticklabels=True)
    plt.savefig(save_name, dpi=600, bbox_inches="tight")
    plt.clf()
    # plt.show()


def precision_with_FPR(trues, pred, prob, FPR=0.05):
    '''
        用于检验模型对unknown cell的探测能力，，不管scGCN还是Seurat等，只要给出了confidence score, 都可以调用这个函数
        prob :query_out.max(dim=1).numpy().reshape(-1)
        细胞属于这个标签的概率
    '''
    unknown_idx = np.where(trues == 'unknown')[0]

    unknown_prob = list(prob[unknown_idx])
    unknown_prob.sort(reverse=True)

    thresh = unknown_prob[int(len(unknown_prob) * FPR)]

    # 取出prob中标记为known的cell
    unknown_idx = np.where(prob < thresh)[0]
    # 把所有小于阈值的标记为unkown
    pred[unknown_idx] = 'unknown'

    true_known_idx = np.where(trues != 'unknown')[0]

    pred = pred[true_known_idx]
    trues = trues[true_known_idx]
    if len(true_known_idx) == 0:
        return 0
    return accuracy_score(trues, pred)


def show_result(ret, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    joint_embedding = np.concatenate([ret['ref_out'], ret['query_out']], axis=0)
    ref_h = joint_embedding[:ret['ref_out'].shape[0], :]
    query_h = joint_embedding[ret['ref_out'].shape[0]:, :]

    # evaluation metrics
    # total_s_score = silhouette_score(joint_embedding, list(ret['ref_label']) + list(ret['pred']))
    # ref_s_score = silhouette_score(ref_h, ret['ref_label'])
    # q_s_score = silhouette_score(query_h, ret['pred'])

    # ari = adjusted_rand_score(ret['query_label'], ret['pred'])
    # bme = batch_mixing_entropy(ref_h, query_h)
    # bme = sum(bme) / len(bme)
    acc = accuracy_score(ret['query_label'], ret['pred'])
    print("pred acc is {:.3f}".format(acc))
    # f1 = f1_score(ret['query_label'], ret['pred'], average='macro')

    '''
        打印每种细胞类型的acc
    '''
    cell_types = set(ret['query_label'])
    for c_t in cell_types:
        print("{:} accuracy is {:.3f}".format(c_t, precision_of_cell(c_t, ret['pred'], ret['query_label'])))

    print("Prediction Accuracy is {:.3f}".format(acc))
    confusion_plot(ret['pred'], ret['query_label'], save_name=os.path.join(save_path, 'pred_confusion_plot.png'))

    return None
    '''
        unknown_cell type的的实验，保存true label，preds和prob
    '''
    # FPR = 0.05
    # print("FPR {:}, precision is {:.3f}".format(FPR, precision_with_FPR(ret['query_label'].copy(), ret['pred'].copy(), ret['prob'].copy(), FPR)))

    # query_trues = pd.DataFrame(data=ret['query_label'], columns=['type'])
    # query_preds = pd.DataFrame(data=ret['pred'], columns=['type'])
    # query_prob = pd.DataFrame(data=ret['prob'], columns=['prob'])
    #
    # query_trues.to_csv(os.path.join(save_path, 'query_labels.csv'), index=False)
    # query_preds.to_csv(os.path.join(save_path, 'query_preds.csv'), index=False)
    # query_prob.to_csv(os.path.join(save_path, 'query_prob.csv'), index_label=False)
    # return
    '''
        展示GCN embeddings
    '''
    # raw_trues = np.concatenate([ret['ref_raw_label'], ret['query_label']]).reshape(-1)
    # gcn_joint_embeddings = np.concatenate([ret['ref_gcn_embeddings'], ret['query_gcn_embeddings']], axis=0)
    # gcn_joint_embeddings_2d = runUMAP(gcn_joint_embeddings)

    # joint_norm_data = np.concatenate([ret['ref_norm_data'], ret['query_norm_data']], axis=0)
    # joint_norm_data_2d = runUMAP(joint_norm_data)
    # show_cluster(joint_norm_data_2d,
    #              ["reference" for i in range(ref_h.shape[0])] + ["query" for i in range(query_h.shape[0])],
    #              "norm_data_ref_query",
    #              save_path)
    # show_cluster(joint_norm_data_2d,
    #              raw_trues,
    #              "norm_data_with_labels",
    #              save_path)

    # show_cluster(gcn_joint_embeddings_2d,
    #              ["reference" for i in range(ref_h.shape[0])] + ["query" for i in range(query_h.shape[0])],
    #              "gcn_embeddings_ref_query",
    #              save_path)
    # show_cluster(gcn_joint_embeddings_2d,
    #              raw_trues,
    #              "gcn_embeddings_with_labels",
    #              save_path)
    # gcn_joint_embeddings_2d = pd.DataFrame(data=gcn_joint_embeddings_2d, columns=['x', 'y'])
    # gcn_label = pd.DataFrame(data=["SeqWell" for i in range(ref_h.shape[0])] + ["10X_V3" for i in range(query_h.shape[0])], columns=['type'])
    # gcn_joint_embeddings_2d.to_csv('gcn_2d.csv', index=False)
    # gcn_label.to_csv('gcn_label.csv', index=False)

    '''
        2023.1.3 以下代码暂时注释掉，为了multi ref更快显示结果，同时保存模型
    '''
    # print('f1-score is {:.3f}'.format(f1))
    # print('Prediction ARI is {:.3f}'.format(ari))
    # print('batch mixing score is {:.3f}'.format(bme))


    # 这部分和原来的feature对应
    # raw_trues = np.concatenate([ret['ref_raw_label'], ret['query_label']]).reshape(-1)
    # # 这部分和h对应
    # trues_after_shuffle = np.concatenate([ret['ref_label'], ret['query_label']]).reshape(-1)
    # all_preds = np.concatenate([ret['ref_label'], ret['pred']]).reshape(-1)
    # raw_data = np.concatenate([ret['ref_raw_data'], ret['query_raw_data']], axis=0)
    #
    # raw_data_2d = runUMAP(raw_data)
    # h_data_2d = runUMAP(joint_embedding)
    #
    # show_cluster(raw_data_2d, raw_trues, 'reference-query raw true label', save_path)
    # show_cluster(h_data_2d, trues_after_shuffle, 'reference-query h true label', save_path)
    # show_cluster(h_data_2d, all_preds, 'reference-query h pred label', save_path)
    # show_cluster(raw_data_2d,
    #              ["reference" for i in range(ref_h.shape[0])] + ["query" for i in range(query_h.shape[0])],
    #              "raw batches",
    #              save_path)
    # show_cluster(h_data_2d,
    #              ["reference" for i in range(ref_h.shape[0])] + ["query" for i in range(query_h.shape[0])],
    #              "h batches",
    #              save_path)
    # #
    query_preds = pd.DataFrame(data=ret['pred'], columns=['type'])
    query_labels = pd.DataFrame(data=ret['query_label'], columns=['type'])
    # all_preds = pd.DataFrame(data=all_preds, columns=['type'])
    # embeddings_2d = pd.DataFrame(data=h_data_2d, columns=['x', 'y'])
    # raw_labels = pd.DataFrame(data=raw_trues, columns=['type'])
    # raw_data_2d = pd.DataFrame(data=raw_data_2d, columns=['x', 'y'])
    #
    #
    #
    query_preds.to_csv(os.path.join(save_path, 'query_preds.csv'), index=False)
    query_labels.to_csv(os.path.join(save_path, 'query_labels.csv'), index=False)
    # all_preds.to_csv(os.path.join(save_path, 'all_preds.csv'), index=False)
    # raw_labels.to_csv(os.path.join(save_path, 'raw_labels.csv'), index=False)
    # embeddings_2d.to_csv(os.path.join(save_path, 'embeddings_2d.csv'), index=False)
    # raw_data_2d.to_csv(os.path.join(save_path, 'raw_data_2d.csv'), index=False)

    '''
        ====== 以上 ======
    '''
    # show_cluster(raw_data_2d[:ret['ref_raw_data'].shape[0],:], ret['ref_raw_label'], "raw reference data", save_path)
    # show_cluster(raw_data_2d[ret['ref_raw_data'].shape[0]:,], ret['query_raw_label'], "raw query data", save_path)
