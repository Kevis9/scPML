import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

'''
    绘制混淆矩阵
'''
## embedding_ours, 取query部分
pred = np.load('preds.npy', allow_pickle=True)[3727:]
save_name = "ours"

### Seurat
# pred = pd.read_csv('seurat_preds.csv')['x'].to_numpy()
# save_name = "Seurat"

### Chetah
# pred = pd.read_csv('chetah_preds.csv')
# pred['x'] = pred['x'].apply(lambda x: "Unassigned" if x.startswith("Node") else x)
# pred = pred['x'].to_numpy()
# save_name = "Chetah"

# ### scmap
# pred = pd.read_csv('../../scmap_preds.csv')
# pred[pred['ref'] == 'unassigned'] = 'Unassigned'
# pred = pred['ref'].to_numpy()
# save_name = "scmap"

'''
==================================================
'''

true = np.load('raw_trues.npy', allow_pickle=True)[3727:]
name = list(set(pred))
name.append("Unassigned")
# 再次去重
name = list(set(name))
name.sort()

name_idx = {}
for i in range(len(name)):
    name_idx[name[i]] = i

confusion_mat = []
# 行是true，只考虑true的部分
for i in range(len(set(true))):
    confusion_mat.append([0 for j in range(len(name))])

pred = list(pred)
true = list(true)

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
print(data_df)
# 将数据倒置过来
data_df = data_df.reindex(index=data_df.index[::-1])

print(data_df.index)
print(data_df.columns)

sns.heatmap(data=data_df, cmap="Blues", cbar=False, xticklabels=False, yticklabels=False)
plt.savefig(save_name, dpi=600, bbox_inches="tight")
plt.show()
