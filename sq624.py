import pandas as pd
from sklearn import preprocessing
data = pd.read_csv('/Users/kevislin/Desktop/sq627/SeoulBikeData.csv', index_col=0)
# 1.确认数据是否有缺省值
# data.isnull()
# 去掉date, holiday还有functional day
data = data.iloc[:,list(range(1,11))]
# 对离散特征进行独热编码
data = pd.get_dummies(data, columns=['Hour', 'Seasons'])

# 对数据进行归一化
data = data.to_numpy()
# data = preprocessing.
