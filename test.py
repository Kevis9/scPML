import pandas as pd

atac_df = pd.read_csv('/Users/kevislin/Desktop/peak_annotation/10x/data/10x_atac_activity_mat.csv', index_col=0).T
data = atac_df.to_numpy()
from utils import reduce_dimension
data_2d = reduce_dimension(data)

data_df = {
        'x':data_2d[:,0],
        'y':data_2d[:,1],
    }
import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(data=data_df, x='x', y='y')
plt.show()
