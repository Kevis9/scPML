import numpy as np

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
    cd4_idx = np.where(labels=='CD4+ T cell')[0]
    total_edges = sm[cd4_idx, :].sum()
    print("CD4+ T cell总共有 {:} 边".format(total_edges//2))
    with_self = 0
    with_others = 0
    for item in types:
        idx = np.where(labels == item)[0]
        # print(sm[cd4_idx, idx].sum())
        num = sm[cd4_idx][:, idx].sum()//2
        mean_num = num / len(cd4_idx)
        if item == 'CD4+ T cell':
            with_self = mean_num
        else:
            with_others += mean_num
        print("每一个节点和 {:} 平均有 {:.3f}边".format(item, mean_num))

    # print("每个节点和自己的边/和别人的边 {:.3f}".format(with_self/with_others))
    return

import pandas as pd
label = pd.read_csv('label_1.csv').to_numpy().squeeze()
sm = pd.read_csv('mat_path_1_1.csv', index_col=0).to_numpy()
check_out_similarity_matrix(sm, label, 5, "")
