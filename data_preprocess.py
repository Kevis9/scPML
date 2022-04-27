
# 获得两个表达矩阵的公共基因
# sharedGeneMatrix(path1, path2)
#
# 生成mouse_vips的label
# df = pd.read_csv('./transfer_across_tissue/label/mouse_VISP_label.csv')
# labels = df['class'].tolist()
# print(set(labels))
# labels = pd.DataFrame(data=labels,columns=['class'])
# labels.to_csv('./mouse_VIPS_label.csv',index=False)
# exit()
#
# mouse_visp:随机取2000个数据和label
# mouse_visp_df = pd.read_csv('./transfer_across_tissue/scData/mouse_VISP.csv',index_col=0)
# labels_df = pd.read_csv('transfer_across_tissue/label/mouse_VISP_label.csv')
#
# num = labels_df.shape[0]
# idx = [i for i in range(num)]
# np.random.shuffle(idx)
#
#
# mouse_visp_df = mouse_visp_df.iloc[idx[:2000],:]
# labels_df = labels_df.iloc[idx[:2000],:]
# mouse_visp_df = pd.DataFrame(data=mouse_visp_df.values, columns=mouse_visp_df.columns)
#
# mouse_visp_df.to_csv('./mouse_VIPS_cut.csv')
# labels_df.to_csv('./mouse_VIPS_label_cut.csv', index=False)
#
# 对mouse_VISP做一个数据清理，把Label为Low Quality和No Class的样本去掉
# mouse_visp_df = pd.read_csv('./transfer_across_tissue/scData/mouse_VISP_cut.csv', index_col=0)
# label = pd.read_csv('./transfer_across_tissue/label/mouse_VIPS_label_cut.csv')
# idx = (label['class'] != 'No Class')&(label['class'] != 'Low Quality')
#
# mouse_visp_df = mouse_visp_df.loc[idx,:]
# mouse_visp_df = pd.DataFrame(data=mouse_visp_df.values, columns=mouse_visp_df.columns)
# label = label.loc[idx,:]
#
# mouse_visp_df.to_csv('./mouse_VISP_cut.csv')
# label.to_csv('./mouse_VISP_label_cut.csv', index=False)
#
# mouse_VISP_label_cut 数字化
# label = pd.read_csv('transfer_across_tissue/label/mouse_VISP_label_cut.csv')
# label_arr = list(set(label['class'].tolist()))
# for i in range(len(label_arr)):
#     label = label.replace(label_arr[i], i+1)
#
# label.to_csv('./mouse_VISP_label_cut_num.csv', index=False)
#
# 读取mouse_pancreas的label并且数字化
# label = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/GSE84133_RAW/GSM2230761_mouse1_umifm_counts.csv')
# label = pd.DataFrame(data=label['assigned_cluster'], columns=['assigned_cluster'])
#
# label_arr = list(set(label['assigned_cluster'].tolist()))
# print(len(label_arr))
# for i in range(len(label_arr)):
#     label = label.replace(label_arr[i], i+1)


# label.to_csv('./mouse1_pancreas_label.csv', index=False)
# exit()
#
# 获取老鼠的基因
# df = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/GSE84133_RAW/GSM2230762_mouse2_umifm_counts.csv', index_col=0)
# del df['barcode']
# del df['assigned_cluster']
# gene_names = df.columns.tolist()
# mouse_gene_names = pd.DataFrame(data=gene_names, columns=['mouse_gene_names'])
# mouse_gene_names.to_csv('./mouse_gene_names.csv', index=False)
# exit()
#
# '''
#     GSE84133:生成mouse2和human3的公共基因矩阵
#     并且label数字化
# '''
# mouse1_df = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/GSE84133_RAW/GSM2230761_mouse1_umifm_counts.csv', index_col=0)
# mouse2_df = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/GSE84133_RAW/GSM2230762_mouse2_umifm_counts.csv', index_col=0)
#
# human1_df = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/GSE84133_RAW/GSM2230757_human1_umifm_counts.csv', index_col=0)
# human2_df = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/GSE84133_RAW/GSM2230758_human2_umifm_counts.csv', index_col=0)
# human3_df = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/GSE84133_RAW/GSM2230759_human3_umifm_counts.csv', index_col=0)
# human4_df = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/GSE84133_RAW/GSM2230760_human4_umifm_counts.csv', index_col=0)
#
#
# common_gene_names = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/transfer_across_species_data/gene_names/commom_gene_names.csv', index_col=0)
#
# # 把数据纵向连接起来
# human_df = pd.concat([human1_df, human2_df, human3_df, human4_df])
# mouse_df = pd.concat([mouse1_df, mouse2_df])
#
# # 查看下各个class的出现次数 (突然觉得pandas内置的函数用的好爽）
# # human_label_counts = human_df.loc[:,"assigned_cluster"].value_counts()
# # mouse_label_counts = mouse_df.loc[:,"assigned_cluster"].value_counts()
#
# cell_type_arr = ['beta', 'alpha', 'ductal', 'acinar', 'delta', 'activated_stellate', 'gamma', 'endothelial', 'quiescent_stellate', 'macrophage', 'mast']
# cell_type_arr.sort()
#
# # 属于cell_type的样本
# human_df = human_df.loc[human_df['assigned_cluster'].isin(cell_type_arr), :]
# mouse_df = mouse_df.loc[mouse_df['assigned_cluster'].isin(cell_type_arr), :]
#
#
# label_num = [i+1 for i in range(len(cell_type_arr))]
# mouse_df['assigned_cluster'].replace(cell_type_arr, label_num, inplace=True) # inplace代表修改原来的df
# human_df['assigned_cluster'].replace(cell_type_arr, label_num, inplace=True)
#
# for i in range(len(cell_type_arr)):
#     mouse_df['assigned_cluster'].replace(cell_type_arr[i], i+1)
#     human_df['assigned_cluster'].replace(cell_type_arr[i], i+1)
#     label_dic[cell_type_arr[i]] = i+1
#
#
# mouse_label_df = pd.DataFrame(data=mouse_df['assigned_cluster'].tolist(), columns=['class'])
# human_label_df = pd.DataFrame(data=human_df['assigned_cluster'].tolist(), columns=['class'])
#
#
# # GSE84133: 生成label, 并进行数字化
# mouse_label_df.to_csv('./mouse_pancreas_label.csv', index=False)
# human_label_df.to_csv('./human_pancreas_label.csv', index=False)
#
#
# # 对原表达矩阵基因和同源基因再取交集
# human_names = human_df.columns.intersection(common_gene_names['human']).tolist()
# idx = common_gene_names['human'].isin(human_names)
#
# common_gene_names = common_gene_names.loc[idx, :]
#
# mouse_df = mouse_df[common_gene_names['mouse']]
# human_df = human_df[common_gene_names['human']]
#
# mouse_df.to_csv('./mouse_pancreas.csv')
# human_df.to_csv('./human_pancreas.csv')
#
# print(mouse_df.shape)
# print(human_df.shape)
# exit()

# mouse_df = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/transfer_across_species_data/scData/mouse_pancreas.csv',index_col=0)
# human_df = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/transfer_across_species_data/scData/human_pancreas.csv',index_col=0)
# print(mouse_df.shape)
# print(human_df.shape)
#
# mouse_std_idx = (mouse_df.std()!=0).tolist()
#
#
# mouse_df = mouse_df.loc[:, mouse_std_idx]
# human_df = human_df.loc[:, mouse_std_idx]
#
# human_std_idx = (human_df.std()!=0).tolist()
#
# mouse_df = mouse_df.loc[:, human_std_idx]
# human_df = human_df.loc[:, human_std_idx]
#
#
# human_label = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/transfer_across_species_data/label/human_pancreas_label.csv')
# mouse_label = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/transfer_across_species_data/label/mouse_pancreas_label.csv')
#
#
#
# h_idx = ((human_label['class'] != 10) & (human_label['class'] != 1)).tolist()
#
# human_df = human_df.loc[h_idx, :]
# human_label = human_label.loc[h_idx, :]
#
# mouse_label.replace([2,3,4,5,6,7,8,9,11],[1,2,3,4,5,6,7,8,9])
# human_label.replace([2,3,4,5,6,7,8,9,11],[1,2,3,4,5,6,7,8,9])
#
# mouse_df.to_csv('mouse_pancreas.csv')
# human_df.to_csv('human_pancreas.csv')
# print(mouse_df.shape)
# print(human_df.shape)
# mouse_label.to_csv('mouse_pancreas_label.csv', index=False)
# human_label.to_csv('human_pancreas_label.csv', index=False)
# print(mouse_label.shape)
# print(human_label.shape)
# exit()



# data = spio.mmread('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/PBMC/counts.read.txt')
# data = data.todense()
# print(data.shape)
# col = np.loadtxt('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/PBMC/cells.read.new.txt', dtype=str)
# print(col.shape)
# row = np.loadtxt('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/PBMC/genes.read.txt', dtype=str)
# print(row.shape)
# data_df = pd.DataFrame(data=data, index=row, columns=col)
# meta_data = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/PBMC/meta_change.txt', sep='\t')
# data_df = data_df.loc[:, meta_data['NAME'].tolist()]
#
#
# def get_df(method):
#     idx = (meta_data['Method'] == method).tolist()
#     df = data_df.iloc[:, idx].T
#     label = meta_data.iloc[idx, :]['CellType']
#     return df, label
#
#
# cel_seq2_df, cel_seq2_label = get_df('CEL-Seq2')
# v3_10x_df, v3_10x_label = get_df('10x Chromium (v3)')
# indrop_df, in_drop_label = get_df('inDrops')
# drop_seq_df, drop_seq_label = get_df('Drop-seq')
# seq_well_df, seq_well_label = get_df('Seq-Well')
#
# def save_as_csv(df, label, name):
#     df.to_csv(name+'_data.csv')
#     label.to_csv(name+'_label.csv',index=False)
#
# save_as_csv(cel_seq2_df, cel_seq2_label, 'CEL_Seq2')
# save_as_csv(v3_10x_df, v3_10x_label, '10X_v3')
# save_as_csv(indrop_df, in_drop_label, 'InDrop')
# save_as_csv(drop_seq_df, drop_seq_label, 'DropSeq')
# save_as_csv(seq_well_df, seq_well_label, 'Seq_Well')
#
# exit()
import numpy as np
import pandas as pd


def get_rid_of_0_gene(df1, df2):
    gene_idx = list(map(lambda x: x[0] & x[1], zip((df1.std() != 0).tolist(), (df2.std() != 0).tolist())))
    df1 = df1.iloc[:, gene_idx]
    df2 = df2.iloc[:, gene_idx]
    return df1, df2


# cel_seq_data = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/PBMC/CEL_Seq2/CEL_Seq2_data.csv', index_col=0)
#
# cel_seq_label = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/PBMC/CEL_Seq2/CEL_Seq2_label.csv')
# data_10x_v3 = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/PBMC/10X_V3/10X_v3_data.csv', index_col=0)
# label_10x_v3 = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/PBMC/10X_V3/10X_v3_label.csv')
#
# cell_type = ['Natural killer cell', 'Cytotoxic T cell', 'CD4+ T cell', 'B cell', 'CD14+ monocyte', 'Megakaryocyte', 'CD16+ monocyte']
#
# idx =label_10x_v3['CellType'].isin(cell_type).tolist()
# data_10x_v3 = data_10x_v3.iloc[idx, :]
# label_10x_v3 = label_10x_v3.iloc[idx, :]
#
# cel_seq_data, data_10x_v3 = get_rid_of_0_gene(cel_seq_data, data_10x_v3)
# cel_seq_label = cel_seq_label.replace(cell_type, [1,2,3,4,5,6,7])
# label_10x_v3 = label_10x_v3.replace(cell_type, [1,2,3,4,5,6,7])
#
# # gene_idx = list(map(lambda x: x[0] & x[1], zip((cel_seq_label.std() != 0).tolist(),(data_10x_v3.std()!=0).tolist())))
# # cel_seq_data = cel_seq_data.iloc[:, gene_idx]
# # data_10x_v3 = data_10x_v3.iloc[:, gene_idx]
#
# gene_names = cel_seq_data.columns.tolist()
# gene_names = [gene.split('_')[1] for gene in gene_names]
#
# cel_seq_data = pd.DataFrame(data=cel_seq_data.values, columns=gene_names, index=cel_seq_data.index.tolist())
# data_10x_v3 = pd.DataFrame(data=data_10x_v3.values, columns=gene_names, index=data_10x_v3.index.tolist())
#
# print(cel_seq_data.shape)
# print(cel_seq_label.shape)
# print(data_10x_v3.shape)
# print(label_10x_v3.shape)
#
#
# cel_seq_data.to_csv('cel_seq2_data.csv')
# cel_seq_label.to_csv('cel_seq2_label.csv', index=False)
#
# data_10x_v3.to_csv('10x_v3_data.csv')
# label_10x_v3.to_csv('10x_v3_label.csv', index=False)
#
# exit()


# 更换mouse的column为human的，并且去掉acti...类型的细胞（此时对应的是1）
# mouse_df = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/transfer_across_species_data/mouse_pancreas.csv', index_col=0)
# human_df = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/transfer_across_species_data/human_pancreas.csv', index_col=0)
#
# mouse_df.columns = human_df.columns
#
# mouse_label = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/transfer_across_species_data/mouse_label.csv')
# human_label = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/transfer_across_species_data/human_label.csv')
#
# # 去除掉1，并且对label进行替换
# mouse_idx = (mouse_label['class'] != 1).tolist()
# mouse_df = mouse_df.iloc[mouse_idx, :]
# mouse_label = mouse_label.iloc[mouse_idx,:]
#
# human_idx = (human_label['class'] != 1).tolist()
# human_df = human_df.iloc[human_idx, :]
# human_label = human_label.iloc[human_idx,:]
#
# mouse_label.replace([2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8], inplace=True)
# human_label.replace([2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8], inplace=True)
#
#
# print(mouse_df.shape)
# print(human_df.shape)
# print(mouse_label.shape)
# print(human_label.shape)
#
#
# mouse_df.to_csv('mouse_data.csv')
# human_df.to_csv('human_data.csv')
# mouse_label.to_csv('mouse_label.csv')
# human_label.to_csv('human_label.csv')
#
# exit()


def get_common_types_df(df1, df2, label1, label2):
    col = 'CellType'
    celltype = list(set(label1[col].tolist()) & set(label2[col].tolist()))
    idx1 = label1[col].isin(celltype).tolist()
    idx2 = label2[col].isin(celltype).tolist()
    df1 = df1.iloc[idx1, :]
    label1 = label1.iloc[idx1, :]
    df2 = df2.iloc[idx2, :]
    label2 = label2.iloc[idx2, :]
    return df1, df2, label1, label2


def process_gene_name(df1, df2):
    # 处理PBMC基因名称
    gene_names = df1.columns.tolist()
    gene_names = [gene.split('_')[1] for gene in gene_names]
    df1.columns = gene_names
    df2.columns = gene_names
    return df1, df2

#
#
# import pandas as pd
# smart_seq_data = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/PBMC/readcounts/Smart_seq2/smart_seq2_data.csv', index_col=0)
# smart_seq_label = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/PBMC/readcounts/Smart_seq2/smart_seq2_label.csv')
#
# seq_well_data = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/PBMC/readcounts/Seq_Well/Seq_Well_data.csv', index_col=0)
# seq_well_label = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/PBMC/readcounts/Seq_Well/Seq_Well_label.csv')
#
# smart_seq_data, seq_well_data, smart_seq_label, seq_well_label = get_common_types_df(smart_seq_data, seq_well_data, smart_seq_label, seq_well_label)
# smart_seq_data, seq_well_data = get_rid_of_0_gene(smart_seq_data, seq_well_data)
# smart_seq_data, seq_well_data = process_gene_name(smart_seq_data, seq_well_data)
# print(smart_seq_data.shape, smart_seq_label.shape)
# print(seq_well_data.shape, seq_well_label.shape)
# smart_seq_data.to_csv('smart_seq_data.csv')
# smart_seq_label.to_csv('smart_seq_label.csv', index=False)
#
# seq_well_data.to_csv('seq_well_data.csv')
# seq_well_label.to_csv('seq_well_label.csv', index=False)
#
#
# # 处理掉cel_seq2_10x_v3部分的数据
# cel_seq2_data = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/platform_data/PBMC/cel_seq2_10x_v3/cel_seq2_data.csv', index_col=0)
# data_10x_v3 = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/platform_data/PBMC/cel_seq2_10x_v3/10x_v3_data.csv', index_col=0)
#
# cel_seq2_data, data_10x_v3 = process_gene_name(cel_seq2_data, data_10x_v3)
# print(cel_seq2_data.shape, data_10x_v3.shape)
# cel_seq2_data.to_csv('cel_seq2_data.csv')
# data_10x_v3.to_csv('10x_v3_data.csv')
#
# # label处理
# cel_seq2_label = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/platform_data/PBMC/cel_seq2_10x_v3/cel_seq2_label.csv')
# label_10x_v3 = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/platform_data/PBMC/cel_seq2_10x_v3/10x_v3_label.csv')
#
# cel_seq2_label.replace([1,2,3,4,5,6,7], ['Natural killer cell','Cytotoxic T cell', 'CD4+ T cell', 'B cell', 'CD14+ monocyte','Megakaryocyte','CD16+ monocyte'], inplace=True)
# label_10x_v3.replace([1,2,3,4,5,6,7], ['Natural killer cell','Cytotoxic T cell', 'CD4+ T cell', 'B cell', 'CD14+ monocyte','Megakaryocyte','CD16+ monocyte'], inplace=True)
# cel_seq2_label.to_csv('cel_seq2_label.csv', index=False)
# label_10x_v3.to_csv('10x_v3_label.csv', index=False)
#
# exit()
#
#
# exit()

'''
    A549数据处理
'''
import scipy.io as spio
import pandas as pd

def process_dup_gene_df(df, gene_name):

    # 通过value_counts的方式拿到重复出现的基因名称
    gene_name = pd.Series(gene_name)
    dup_gene_names = pd.value_counts(gene_name)[pd.value_counts(gene_name)>1].index.tolist()
    cell_to_del = set()


    for gene in dup_gene_names:
        tmp_df = df.loc[:, gene]
        cell_idx = tmp_df.apply(lambda x: len(x.unique()), axis=1)
        cell_to_del.update(set(cell_idx[cell_idx > 1].index.tolist()))
        # # 获取那些要删除的细胞（在重复基因中出现表达不一致）
        # for cell in tmp_df.index.tolist():
        #     if len(tmp_df.loc[cell, :].unique()) > 1:
        #         cell_to_del.append(cell)

    cell_to_del = list(cell_to_del)
    print(len(cell_to_del))
    df = df.loc[~df.index.isin(cell_to_del), :]
    # 改变df的columns命名: 如果出现重复的情况，重复的后面跟上次数
    map = dict()
    gene_col = []
    for gene in gene_name:
        if gene in map.keys():
            map[gene] += 1
            gene_col.append(gene+str(map[gene]))
        else:
            gene_col.append(gene)
            map[gene] = 0
    df.columns = gene_col
    return df


def equality_control_for_A549():
    atac_gene = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/A549/ATAC/supplementary/atac_gene.csv',
                            sep='\t')

    rna_gene = pd.read_csv(
        '/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/A549/RNA/GSM3271040_RNA_sciCAR_A549_gene.txt')



    atac_cell = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/A549/ATAC/GSM3271041_ATAC_sciCAR_A549_cell.txt')
    rna_cell = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/A549/RNA/GSM3271040_RNA_sciCAR_A549_cell.txt')

    # common_gene = list(set(atac_gene['gene_name']) & set(rna_gene['gene_short_name']))
    common_gene = list((set(
        atac_gene['gene_name'].value_counts()[atac_gene['gene_name'].value_counts() == 1].index.tolist())) & (set(rna_gene['gene_short_name'].value_counts()[rna_gene['gene_short_name'].value_counts()==1].index.tolist())))


    print(len(common_gene))
    atac_data = spio.mmread(
        '/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/A549/ATAC/GSM3271041_ATAC_sciCAR_A549_peak_count.txt')
    atac_data = atac_data.todense().T


    rna_data = spio.mmread(
        '/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/A549/RNA/GSM3271040_RNA_sciCAR_A549_gene_count.txt')
    rna_data = rna_data.todense().T

    atac_df = pd.DataFrame(data=atac_data, index=atac_cell['sample'].to_list())
    atac_df = atac_df.iloc[:, list(np.array(atac_gene['id'].to_list())-1)]
    atac_df.columns = atac_gene['gene_name'].to_list()

    rna_df = pd.DataFrame(data=rna_data, index=rna_cell['sample'].to_list(), columns=rna_gene['gene_short_name'].to_list())

    common_cell = list(set(atac_df.index.tolist()) & set(rna_df.index.tolist()))


    # 取公共基因和公共cell
    atac_df = atac_df.loc[common_cell, common_gene]
    rna_df = rna_df.loc[common_cell, common_gene]

    print(atac_df.shape)
    print(rna_df.shape)

    # 处理重复的gene(去掉部分异常细胞，更改重复gene名称）

    # atac_df = process_dup_gene_df(atac_df, atac_df.columns.to_list())
    # rna_df = process_dup_gene_df(rna_df, rna_df.columns.to_list())
    # 这里打算直接去除掉重复的gene


    #再取一次公共细胞（因为删除了部分细胞）和公共基因（因为改了重复的名字）
    #  common_cell = list(set(atac_df.index.tolist()) & set(rna_df.index.tolist()))
    #
    # atac_df = atac_df.loc[common_cell, common_gene]
    # rna_df = rna_df.loc[common_cell, common_gene]

    # 去掉NA的数据
    rna_cell = pd.DataFrame(data=rna_cell.values, index=rna_cell['sample'].to_list(), columns=rna_cell.columns)
    rna_cell = rna_cell.loc[common_cell, :]
    not_null_idx = (pd.notnull(rna_cell['treatment_time'])).to_list()

    rna_cell = rna_cell.iloc[not_null_idx, :]
    rna_df = rna_df.iloc[not_null_idx, :]
    atac_df = atac_df.iloc[not_null_idx, :]

    # 两者的label 都是一样的
    label = rna_cell['treatment_time']

    # 去掉方差为0的基因
    rna_df, atac_df = get_rid_of_0_gene(rna_df, atac_df)

    # 保存数据
    print(rna_df.shape)
    print(atac_df.shape)
    print(label.shape)
    rna_df.to_csv('rna_data.csv')
    atac_df.to_csv('atac_data.csv')
    label.to_csv('label.csv', index=False)


# equality_control_for_A549()


'''
    将A549peak数据的三列提取出来
'''
# atac_peak = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/A549/ATAC/GSM3271041_ATAC_sciCAR_A549_peak.txt')
# atac_peak = atac_peak[['chr','start','end','id']]
# arr = ['chr{}'.format(str(i)) for i in atac_peak['chr']]
#
# atac_peak['chr'] = arr
# atac_peak.to_csv('chr_data.csv', sep='\t', index=False)
# exit()




'''
    Kidney data处理
'''
# import scipy.io as spio
# import pandas as pd
# # cell_df = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/Kidney_data/RNA/GSM3271044_RNA_mouse_kidney_cell.txt')
# # gene_df = pd.read_csv('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/Kidney_data/RNA/GSM3271044_RNA_mouse_kidney_gene.txt')
# rna_data = spio.mmread('/Users/kevislin/Desktop/单细胞/资料汇总/data/RAW_data/Kidney_data/RNA/GSM3271044_RNA_mouse_kidney_gene_count.txt')
# rna_data = rna_data.todense()
# rna_data = rna_data.T
# print(rna_data.shape)