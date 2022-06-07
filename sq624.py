import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/kevislin/Desktop/sq627/SeoulBikeData.csv', index_col=0)
# 1.确认数据是否有缺省值
# data.isnull()
# 去掉date, holiday还有functional day

data = data.iloc[:,list(range(1,11))]
# 对离散特征进行独热编码
data = pd.get_dummies(data, columns=['Hour', 'Seasons'])

# 对数据进行归一化(z_score)
data = data.to_numpy()
data = preprocessing.StandardScaler().fit_transform(data)

sse_arr = []
# 直接进行K-means聚类
for k in range(3, 15):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data)
    sse_arr.append(kmeans.inertia_)
    print(sse_arr)

plt.figure()
plt.plot(np.array(list(range(3, 15))), sse_arr)
plt.show()




# kmeans.predict([[0, 0], [12, 3]])

# kmeans.cluster_centers_

