import os.path
import torch
from torch import embedding, nn
from utils import sc_normalization, mask_data, construct_graph, \
    read_data_label_h5, read_similarity_mat_h5, \
    cpm_classify, z_score_normalization, show_cluster, \
    concat_views, batch_mixing_entropy, runPCA, runUMAP, encode_label
from model import scGNN, CPMNets, Classifier
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score
import wandb

# seq_well_smart 只有五类!!!
# drop_seq_10x_v3有8类

# 训练scGNN，得到每个Pathway的embedding
def train_scGCN(model, G_data, optimizer,
                index_pair, masking_idx, norm_data, loss_title):
    '''
    :param model: 待训练的模型
    :param n_epochs:
    :param G_data: 训练的图数据
    :param optimizer:
    :param index_pair: 做过mask元素的index pair
    :param masking_idx: mask元素的index
    :param norm_data: mask之后的norm_data
    :return:
    '''
    model = model.to(device)
    for epoch in range(parameter_config['epoch_GCN']):
        model.train()
        optimizer.zero_grad()
        pred = model(G_data.to(device))

        # 得到预测的droout
        dropout_pred = pred[index_pair[0][masking_idx], index_pair[1][masking_idx]]
        dropout_true = norm_data[index_pair[0][masking_idx], index_pair[1][masking_idx]]
        loss_fct = nn.MSELoss()
        loss = loss_fct(dropout_pred.view(1, -1), torch.tensor(dropout_true, dtype=torch.float).to(device).view(1, -1))
        wandb.log({
            loss_title: loss.item()
        })
        loss.backward()
        optimizer.step()
        if epoch % 200 == 0:
            print('Epoch: {}, Training Loss {:.4f}'.format(epoch, loss.item()))
    return model


def self_supervised_train(data):
    # 可以试试调整这个mask比例来调参
    masked_prob = min(len(data.nonzero()[0]) / (data.shape[0] * data.shape[1]), parameter_config["mask_rate"])
    masked_data, index_pair, masking_idx = mask_data(data, masked_prob)

    sm_arr = [read_similarity_mat_h5(data_config['data_path'], "ref/sm_" + str(i + 1)) for i in
              range(4)]

    graphs = [construct_graph(masked_data, sm_arr[i], parameter_config['k']) for i in range(len(sm_arr))]

    embeddings_arr = []
    gcn_models = []

    # 训练ref data in scGNN
    for i in range(len(graphs)):
        model = scGNN(graphs[i], parameter_config['middle_out'])
        optimizer = torch.optim.Adam(model.parameters())
        model = train_scGCN(model, graphs[i], optimizer, index_pair, masking_idx,
                            data, 'GCN: view' + str(i + 1) + ' loss')

        # 利用未mask的矩阵，构造图，丢入训练好的model，得到中间层embedding
        # embedding = model.get_embedding(Graph(scDataNorm, similarity_matrix_arr[i]))
        # 还是用mask好的数据得到Embedding比较符合自监督的逻辑
        embedding = model.get_embedding(graphs[i])
        embeddings_arr.append(embedding.detach().cpu().numpy())
        gcn_models.append(model)
    return embeddings_arr, gcn_models


def train_query(gcn_models, cpm_model, query_data):
    # 数据预处理
    query_norm_data = sc_normalization(query_data)

    # 构造Query data的Graph
    query_sm_arr = [read_similarity_mat_h5(data_config['data_path'], data_config['query_key']+"/sm_" + str(i + 1)) for i in
                    range(4)]
    query_graphs = [construct_graph(query_norm_data, query_sm_arr[i], parameter_config['k'])
                    for i in range(len(query_sm_arr))]

    # 获得Embedding
    query_views = []
    for i in range(len(gcn_models)):
        query_views.append(gcn_models[i].get_embedding(query_graphs[i]).detach().cpu().numpy())

    query_data_embeddings_tensor = torch.from_numpy(z_score_normalization(concat_views(query_views))).float().to(device)

    # query_data_embeddings_tensor = torch.from_numpy(concat_views(query_views)).float().to(device)
    # query_label_tensor = torch.from_numpy(query_labels).view(1, query_labels.shape[0]).long().to(device)

    query_h = cpm_model.train_query_h(query_data_embeddings_tensor, parameter_config['epoch_CPM_test'])
    return query_h.detach().cpu().numpy()


def transfer_labels():
    if parameter_config['model_exist']:
        # 如果模型存在, 加载已存在的模型，然后对query进行重构
        gcn_models, cpm_model = load_models()
        query_data, query_label = read_data_label_h5(data_config['data_path'], data_config['query_key'])
        query_data = query_data.astype(np.float64)

        ref_h = cpm_model.get_h_train()
        ref_label = cpm_model.get_ref_labels()

        ref_label, query_label, enc = encode_label(ref_label, query_label)

        query_h = train_query(gcn_models, cpm_model, query_data)

        pred = cpm_classify(ref_h, query_h, ref_label)
        acc = (pred == query_label).sum()
        acc = acc / pred.shape[0]

        # 还原label
        ref_label = enc.inverse_transform(ref_label)
        query_label = enc.inverse_transform(query_label)
        pred = enc.inverse_transform(pred)

        ret = {
            'acc': acc,
            'ref_h': ref_h,
            'query_h': query_h,
            'ref_label': ref_label,
            'query_raw_data': query_data,
            'query_label': query_label,
            'pred': pred,
        }
        return ret


    '''
        数据准备
    '''
    # data_path = os.path.join(data_config['data_path'], 'data.h5')

    ref_data, ref_label = read_data_label_h5(data_config['data_path'], "ref")
    ref_data = ref_data.astype(np.float64)

    query_data, query_label = read_data_label_h5(data_config['data_path'], data_config['query_key'])
    query_data = query_data.astype(np.float64)

    # 将ref和query data进行编码
    ref_label, query_label, enc = encode_label(ref_label, query_label)

    # 数据预处理
    ref_norm_data = sc_normalization(ref_data)
    ref_views, ref_gcn_models = self_supervised_train(ref_norm_data)

    '''
        Reference Data Embeddings  
    '''
    # view的个数
    ref_view_num = len(ref_views)
    # 每个view的特征长度
    ref_view_feat_len = []
    for i in range(ref_view_num):
        ref_view_feat_len.append(ref_views[i].shape[1])

    # 把所有的view连接在一起, 并做一个归一化
    ref_data_embeddings_tensor = torch.from_numpy(z_score_normalization(concat_views(ref_views))).float().to(device)
    ref_label_tensor = torch.from_numpy(ref_label).view(1, ref_label.shape[0]).long().to(device)

    '''
        对ref data进行训练
    '''
    cpm_model = CPMNets(ref_view_num,
                        ref_data.shape[0],
                        ref_view_feat_len,
                        enc.inverse_transform(ref_label),
                        parameter_config)

    # 开始训练
    cpm_model.train_ref_h(ref_data_embeddings_tensor, ref_label_tensor)

    # 得到最后的embeddings (ref和query)
    ref_h = cpm_model.get_h_train()
    query_h = train_query(ref_gcn_models, cpm_model, query_data)

    pred = cpm_classify(ref_h, query_h, ref_label)
    acc = (pred == query_label).sum()
    acc = acc / pred.shape[0]

    # 还原label
    ref_label = enc.inverse_transform(ref_label)
    query_label = enc.inverse_transform(query_label)

    pred = enc.inverse_transform(pred)

    ret = {
        'acc': acc,
        'ref_h': ref_h,
        'query_h': query_h,
        'ref_raw_data': ref_data,
        'ref_label': ref_label,
        'query_raw_data': query_data,
        'query_label': query_label,
        'pred': pred,
        'cpm_model': cpm_model,
        'gcn_models': ref_gcn_models
    }
    return ret


def show_result(ret):
    embedding_h = np.concatenate([ret['ref_h'], ret['query_h']], axis=0)
    # embedding_h_pca = runPCA(embedding_h)

    ref_h = embedding_h[:ret['ref_h'].shape[0], :]
    query_h = embedding_h[ret['ref_h'].shape[0]:, :]

    # evaluation metrics
    total_s_score = silhouette_score(embedding_h, list(ret['ref_label']) + list(ret['pred']))
    ref_s_score = silhouette_score(ref_h, ret['ref_label'])
    q_s_score = silhouette_score(query_h, ret['pred'])

    ari = adjusted_rand_score(ret['query_label'], ret['pred'])
    bme = batch_mixing_entropy(ref_h, query_h)
    bme = sum(bme) / len(bme)

    print("Prediction Accuracy is {:.3f}".format(ret['acc']))
    # print('Prediction Silhouette score is {:.3f}'.format(s_score))
    print('Prediction ARI is {:.3f}'.format(ari))

    # 数据上报
    wandb.log({
        'Prediction Acc': ret['acc'],
        'ref Silhouette ': ref_s_score,
        'query Silhouette ': q_s_score,
        'total Silhouette ': total_s_score,
        'ARI': ari,
        'Batch Mixing Entropy Mean': bme
    })

    all_true_labels = np.concatenate([ret['ref_label'], ret['query_label']]).reshape(-1)
    all_pred_labels = np.concatenate([ret['ref_label'], ret['pred']]).reshape(-1)

    if not parameter_config['model_exist']:
        raw_data = np.concatenate([ret['ref_raw_data'], ret['query_raw_data']], axis=0)
        # raw_data_pca = runPCA(raw_data)
        #
        raw_data_2d = runUMAP(raw_data)  # 对PCA之后的数据进行UMAP可视化
        # ref_len = ret['ref_raw_data'].shape[0]
        #
        np.save('result/raw_data_2d.npy', raw_data_2d)
        show_cluster(raw_data_2d, all_true_labels, 'reference-query raw true label')

    h_data_2d = runUMAP(embedding_h)
    np.save('result/h_data_2d.npy', h_data_2d)
    np.save('result/all_true_labels.npy', all_true_labels)
    np.save('result/all_pred_labels.npy', all_pred_labels)


    show_cluster(h_data_2d, all_true_labels, 'reference-query h true label')
    show_cluster(h_data_2d, all_pred_labels, 'reference-query h pred label')

    # show_cluster(raw_data_2d[:ref_len, :], ret['ref_label'], 'Raw reference data')
    # show_cluster(raw_data_2d[ref_len:, :], ret['query_label'], 'Raw query data')
    # show_cluster(h_data_2d[:ref_len, :], ret['ref_label'], 'Reference h')
    # show_cluster(h_data_2d[ref_len:, :], ret['query_label'], 'Query h')
    # show_cluster(h_data_2d[ref_len:, :], ret['pred'], 'Query h with prediction label')

    # For multi omics part
    # show_cluster(h_data_2d, np.concatenate([['Reference' for i in range(len(ret['ref_label']))], ['Query' for i in range(len(ret['query_label']))]])
    #              , 'Reference-Query H')
    # show_cluster(h_data_2d, np.concatenate([ret['ref_label'].reshape(-1), ret['query_label'].reshape(-1)])
    #              , 'Reference-Query H with pred label')
    # show_cluster(raw_data_2d, np.concatenate([['Reference' for i in range(len(ret['ref_label']))], ['Query' for i in range(len(ret['query_label']))]])
    #              , 'Reference-Query Raw')

def save_models(gcn_models, cpm_model):
    for i, model in enumerate(gcn_models):
        torch.save(model, 'result/gcn_model_'+ str(i) + '.pt')
    torch.save(cpm_model, 'result/cpm_model.pt')



def load_models():
    gcn_models = []
    for i in range(4):
        model = torch.load('result/gcn_model_'+ str(i) + '.pt')
        gcn_models.append(model)
    cpm_model = torch.load('result/cpm_model.pt')

    return gcn_models, cpm_model




def main_process():
    run = wandb.init(project="cell_classify_" + data_config['project'],
                     entity="kevislin",
                     config={"config": parameter_config, "data_config": data_config},
                     tags=[data_config['ref_name'] + '-' + data_config['query_name'], data_config['project']],
                     reinit=True)
    ret = transfer_labels()

    # 保存模型, 利用pickle
    if not parameter_config['model_exist'] and ret['acc']>0.95:
        save_models(ret['gcn_models'], ret['cpm_model'])
        exit()
    # 查看结果
    show_result(ret)

    run.finish()

# 数据配置
data_config = {
    'data_path': 'F:\\yuanhuang\\kevislin\\data\\species\\task1\\data.h5',
    'ref_name': 'GSE84133: mouse',
    'query_name': 'E_MTAB_5061: human',
    # 'query_name': 'GSE84133: human',
    'query_key': 'query/query_1',
    'project': 'species',
    'ref_class_num': 8,
    'dataset_name': 'GSE84133',
}
# {'gamma', 'alpha', 'endothelial', 'macrophage', 'ductal', 'delta', 'beta', 'quiescent_stellate'}

parameter_config = {
    'ref_class_num': data_config['ref_class_num'],  # Reference data的类别数
    'epoch_GCN': 2500,  # Huang model 训练的epoch
    'epoch_CPM_train': 3000,
    'epoch_CPM_test': 3000,
    'lsd_dim': 128,  # CPM_net latent space dimension
    'k': 2,  # 图构造的时候k_neighbor参数
    'middle_out': 2048,  # GCN中间层维数
    'w_classify': 10,  # classfication loss的权重
    'mask_rate': 0.3,
    'model_exist': True,  # 如果事先已经有了模型,则为True
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Transfer across " + data_config['project'])
print("Reference: " + data_config['ref_name'], "Query: " + data_config['query_name'])

# 测试epoch_CPM_train
# main_process(data_config, config)
main_process()
main_process()
main_process()






