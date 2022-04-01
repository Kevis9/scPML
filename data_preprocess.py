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


#
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
# exit()