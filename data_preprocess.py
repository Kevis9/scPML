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
