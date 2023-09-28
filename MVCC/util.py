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
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

def mean_norm(data):
    '''    
    :param data: ndarray (cells * genes)
    :return: 返回归一化的矩阵
    '''
    row_sum = np.sum(data, axis=1)
    mean_transcript = np.mean(row_sum)        
    row_sum[np.where(row_sum == 0)] = 1

    scale_factor = 1e4
    # data_norm = np.log1p((data / row_sum.reshape(-1 ,1))*scale_factor)
    data_norm = (data / row_sum.reshape(-1, 1)) * mean_transcript

    return data_norm


def z_score_scale(data):    
    means = np.mean(data, axis=0)
    standard = np.std(data, axis=0)

    return (data - means) / standard


def mask_data(data, masked_prob):
    '''
    :param data: expression data (cells * genes)
    :param masked_prob: mask probability
    :return:
        1. X: masked data
        2. index_pair: [(row),(col)],
        3. masking_idx: index of masked data
        X[index_pair[0][masking_idx], index_pair[1][masking_idx]] is the masked data
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
    random.seed(seed)  
    idx = random.sample(range(index_pair[0].shape[0]), int(index_pair[0].shape[0] * masked_prob))  # 无重复采样
    mask_rows = index_pair[0][idx]
    
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
    
    for (u, v) in G.edges():
        edges.append([u, v])
        edges.append([v, u])
    edges = np.array(edges).T
    edges = torch.tensor(edges, dtype=torch.long)
    feat = torch.tensor(data, dtype=torch.float)
    
    g_data = geoData(x=feat, edge_index=edges)
    return g_data


def check_out_similarity_matrix(sm, labels, k, sm_name):
    
    sm = construct_adjacent_matrix_with_MNN(sm, k)
    types = sorted(list(set(labels)))
    
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
    
    confusion_mat = np.array(confusion_matrix)    
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
    A = kneighbors_graph(data, k, mode='connectivity', include_self=False)  # 拿到Similarity矩阵
    return A.todense()

def construct_adjacent_matrix(similarity_mat, k):    
    if k==0:
        '''
            back to a FC network if k==0
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
    k_idxs = []    
    similarity_mat[np.diag_indices_from(similarity_mat)] = 0
    for i in range(similarity_mat.shape[0]):
        top_k_idx = similarity_mat[i].argsort()[::-1][0:k]
        k_idxs.append(top_k_idx)
    adjacent_mat = np.zeros(shape=similarity_mat.shape)
    for i in range(similarity_mat.shape[0]):
        adjacent_mat[i, k_idxs[i]] = 1

    # should contain each other
    adjacent_mat = np.logical_and(adjacent_mat, adjacent_mat.T)
    adjacent_mat = adjacent_mat.astype(np.int64)
    
    return adjacent_mat

def construct_graph(data, sm_mat, k):

    sm_mat = construct_adjacent_matrix_with_MNN(sm_mat, k)
    # sm_mat = construct_adjacent_matrix(sm_mat, k)

    graph = nx.from_numpy_matrix(np.matrix(sm_mat))

    edges = []
    
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

    print('data shape is :{}'.format(data.shape))  # (samples,genes)
    print('label shape is : {}'.format(label.shape))
    return data, label




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
        
    sel_model = SelectKBest(k=nf)  # default score function is f_classif

    sel_model.fit(data1, label1)
    idx = sel_model.get_support(indices=True)
    
    # idx = list(set(idx1) & set(idx2))
    data1 = data1[:, idx]
    data2 = data2[:, idx]
    print("After gene selction , ref data shape {:}, query data shape {:}".format(data1.shape, data2.shape))

    return data1, data2



def pre_process(data1, data2, label1, nf=2000):
    
    data1, data2 = sel_feature(data1, data2, label1, nf=nf)
    # scaler = StandardScaler()
    data1 = mean_norm(data1)
    data2 = mean_norm(data2)
    # data1 = scaler.fit_transform(data1)
    # data2 = scaler.transform(data2)    
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
    visualize clusters
    :param data: 2-d data
    :param label: 
    :param title: title for plot
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
    # df = df[~((df['x']>10) | (df['y']>10))]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='x', y='y', hue='label', palette='deep', s=3)
    plt.legend(loc=3, bbox_to_anchor=(1, 0))  
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.title(title)
    plt.savefig(os.path.join(save_path, "_".join(title.split()) + '.png'), bbox_inches='tight')  # dpi可以设置
    


def concat_views(views):
    return np.concatenate(views, axis=1).astype(np.float64)


def batch_mixing_entropy(ref_data, query_data, L=100, M=300, K=500):
    '''
    :param ref_data:
    :param query_data:
    :param L: times
    :param M: number of randomly sampling cells
    :param k: number of neigbors
    :return: list of batch entropy, representing results of L randomly sampling
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
    
    confusion_mat = np.array(confusion_mat)    
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

    
    cell_types = set(ret['query_label'])
    for c_t in cell_types:
        print("{:} accuracy is {:.3f}".format(c_t, precision_of_cell(c_t, ret['pred'], ret['query_label'])))

    print("Prediction Accuracy is {:.3f}".format(acc))
    confusion_plot(ret['pred'], ret['query_label'], save_name=os.path.join(save_path, 'pred_confusion_plot.png'))

        
    # 这部分和原来的feature对应
    raw_trues = np.concatenate([ret['ref_raw_label'], ret['query_label']]).reshape(-1)
    # 这部分和h对应
    trues_after_shuffle = np.concatenate([ret['ref_label'], ret['query_label']]).reshape(-1)
    all_preds = np.concatenate([ret['ref_label'], ret['pred']]).reshape(-1)
    raw_data = np.concatenate([ret['ref_raw_data'], ret['query_raw_data']], axis=0)
    
    raw_data_2d = runUMAP(raw_data)
    h_data_2d = runUMAP(joint_embedding)
    
    show_cluster(raw_data_2d, raw_trues, 'reference-query raw true label', save_path)
    show_cluster(h_data_2d, trues_after_shuffle, 'reference-query h true label', save_path)
    show_cluster(h_data_2d, all_preds, 'reference-query h pred label', save_path)
    show_cluster(raw_data_2d,
                 ["reference" for i in range(ref_h.shape[0])] + ["query" for i in range(query_h.shape[0])],
                 "raw batches",
                 save_path)
    show_cluster(h_data_2d,
                 ["reference" for i in range(ref_h.shape[0])] + ["query" for i in range(query_h.shape[0])],
                 "h batches",
                 save_path)
    
    query_preds = pd.DataFrame(data=ret['pred'], columns=['type'])
    query_labels = pd.DataFrame(data=ret['query_label'], columns=['type'])
    all_preds = pd.DataFrame(data=all_preds, columns=['type'])
    embeddings_2d = pd.DataFrame(data=h_data_2d, columns=['x', 'y'])
    raw_labels = pd.DataFrame(data=raw_trues, columns=['type'])
    raw_data_2d = pd.DataFrame(data=raw_data_2d, columns=['x', 'y'])
    query_preds.to_csv(os.path.join(save_path, 'query_preds.csv'), index=False)
    query_labels.to_csv(os.path.join(save_path, 'query_labels.csv'), index=False)
    all_preds.to_csv(os.path.join(save_path, 'all_preds.csv'), index=False)
    raw_labels.to_csv(os.path.join(save_path, 'raw_labels.csv'), index=False)
    embeddings_2d.to_csv(os.path.join(save_path, 'embeddings_2d.csv'), index=False)
    raw_data_2d.to_csv(os.path.join(save_path, 'raw_data_2d.csv'), index=False)

