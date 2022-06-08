import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import time

def sse_cal(data, label):
    label_num = len(set(label.tolist()))
    print(set(label))
    center_mean = [data[label == i].mean() for i in range(label_num)]
    sse = 0
    for point, label in zip(data, label):
        # print(point)
        # print(center_mean[label])
        sse += np.square(point - center_mean[label]).sum()
        print(sse)
    return sse

def check_csm_kmeans(data):
    # 直接进行K-means聚类
    csm_arr = []
    for k in range(2, 30):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data)
        s = silhouette_score(data, kmeans.labels_)
        csm_arr.append(s)
    plt.figure()
    plt.title('SeoulBikeData')
    plt.xlabel('K')
    plt.ylabel('CSM')
    plt.plot(np.array(list(range(2, 30))), csm_arr, '-bx')
    plt.show()

def get_metrics(data):
    t_b = time.time()
    kmeans = KMeans(n_clusters=10, random_state=0)
    kmeans.fit(data)
    t_e = time.time()
    time_taken = t_e - t_b
    s_score = silhouette_score(data, kmeans.labels_)
    sse = sse_cal(data, kmeans.labels_)
    return time_taken, sse, s_score


def main_procees(data_name, path):
    if(data_name=='1'):
        # Data1: SeoulBikeData
        data = pd.read_csv(path, index_col=0)
        # 1.确认数据是否有缺省值
        # 去掉date, holiday还有functional day
        data = data.iloc[:, list(range(1, 11))]
        # 对离散特征进行独热编码
        data = pd.get_dummies(data, columns=['Hour', 'Seasons'])

        # 对数据进行归一化(z_score)
        data = data.to_numpy()
        data = preprocessing.StandardScaler().fit_transform(data)
        # 利用PCA对数据进行降维操作
        data = PCA(n_components=5).fit_transform(data)
        # check_csm_kmeans(data)
        # check_csm_dbscan(data)
        print("(time_taken, sse, silhouette score) : {}".format(get_metrics(data)))
    if(data_name=='2'):
        # Data2: Facebook_Live_sellers
        data = pd.read_csv(path, index_col=0)
        data = data.iloc[:, [0]+list(range(2, 10))]

        # 保留为photo和video, link的type
        idx = data['status_type'].isin(['photo', 'video', 'link']).tolist()
        data = data.iloc[idx, :]
        data = pd.get_dummies(data, columns=['status_type'])

        # 对数据进行归一化(z_score)
        data = data.to_numpy()
        data = preprocessing.StandardScaler().fit_transform(data)

        data = PCA(n_components=5).fit_transform(data)

        check_csm_kmeans(data)
        print("(time_taken, sse, silhouette score) : {}".format(get_metrics(data)))
    if(data_name=='3'):
        # Data2: Facebook_Live_sellers
        data = pd.read_csv(path, header=None)
        data = data.iloc[:, 1:]

        # 对数据进行清洗，将?转为0
        data = data.apply(lambda x : x.replace('?', 0))

        # 对数据进行归一化(z_score)
        data = data.to_numpy()
        data = preprocessing.StandardScaler().fit_transform(data)
        data = PCA(n_components=5).fit_transform(data)

        check_csm_kmeans(data)
        print("(time_taken, sse, silhouette score) : {}".format(get_metrics(data)))

main_procees("1", 'C:\\Users\\Lenovo\\Desktop\\sq627\\SeoulBikeData.csv')