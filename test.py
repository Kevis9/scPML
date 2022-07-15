from utils import sc_normalization,\
    read_data_label_h5, read_similarity_mat_h5, encode_label, show_result
from model import  MVCCModel
import numpy as np
import wandb

# 数据配置
data_config = {
    'data_path': 'F:\\yuanhuang\\kevislin\\data\\species\\task1\\data.h5',
    'ref_name': 'GSE84133: mouse',
    # 'query_name': 'E_MTAB_5061: human',
    'query_name': 'GSE84133: human',
    'ref_key' : 'ref',
    'query_key': 'query/query_1',
    'project': 'species',
}
# ['gamma', 'alpha', 'endothelial', 'macrophage', 'ductal', 'delta', 'beta', 'quiescent_stellate']

parameter_config = {
    # GCN部分
    'epoch_gcn': 3000,  # Huang model 训练的epoch
    'k': 2,  # 图构造的时候k_neighbor参数
    'middle_out': 4096,  # GCN中间层维数
    'mask_rate': 0.3,
    # CPM 部分
    'epoch_CPM_train': 500,
    'epoch_CPM_test': 1000,
    'batch_size_cpm': 256,  # CPM中重构和分类的batch size
    'lsd': 2049,  # CPM_net latent space dimension
    'lamb': 1,  # classfication loss的权重
    'model_exist_gcn': True,  # 如果事先已经有了模型,则为True
    'model_exist_cpm': True,
}



def main_process():
    run = wandb.init(project="cell_classify_" + data_config['project'],
                     entity="kevislin",
                     config={"config": parameter_config, "data_config": data_config},
                     tags=[data_config['ref_name'] + '-' + data_config['query_name'], data_config['project']],
                     reinit=True)


    # 数据准备
    ref_data, ref_label = read_data_label_h5(data_config['data_path'], data_config['ref_key'])
    query_data, query_label = read_data_label_h5(data_config['data_path'], data_config['query_key'])
    ref_data = ref_data.astype(np.float64)
    query_data = query_data.astype(np.float64)
    # 将ref和query data进行编码
    ref_label, query_label, enc = encode_label(ref_label, query_label)

    # 数据预处理 (mask部分放在MVCC Model里面进行处理）
    ref_norm_data = sc_normalization(ref_data)
    query_norm_data = sc_normalization(query_data)

    ref_sm_arr = [read_similarity_mat_h5(data_config['data_path'], data_config['ref_key']+"/sm_" + str(i + 1)) for i in
              range(4)]
    query_sm_arr = [read_similarity_mat_h5(data_config['data_path'], data_config['query_key'] + "/sm_" + str(i + 1)) for
                    i in
                    range(4)]


    mvvcmodel = MVCCModel(cpm_exist=parameter_config['model_exist_cpm'],
                          gcn_exist=parameter_config['model_exist_gcn'],
                          gcn_middle_out=parameter_config['middle_out'],
                          gcn_input_dim=ref_data.shape[1],
                          lsd=parameter_config['lsd'],
                          class_num=len(set(ref_label)),
                          view_num=4,
                          epoch_gcn=parameter_config['epoch_gcn'],
                          k_neighbor=parameter_config['k'],
                          epoch_cpm_train=parameter_config['epoch_CPM_train'],
                          epoch_cpm_test=parameter_config['epoch_CPM_test'],
                          batch_size_cpm=parameter_config['batch_size_cpm'],
                          lamb=parameter_config['lamb'])
    # mvvcmodel.fit(ref_norm_data, ref_sm_arr, ref_label)

    pred = mvvcmodel.predict(query_norm_data, query_sm_arr)

    ref_out, query_out = mvvcmodel.get_embeddings()
    ref_out = ref_out.detach().cpu().numpy()
    query_out = query_out.detach().cpu().numpy()
    print((pred==query_label).sum()/pred.shape[0])

    exit()
    ret = {
        'ref_out': ref_out,
        'query_out': query_out,
        'ref_raw_data': ref_data,
        'ref_label': ref_label,
        'query_raw_data': query_data,
        'query_label': query_label,
        'pred': pred,
    }
    show_result(ret)

    run.finish()


main_process()

