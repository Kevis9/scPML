import pandas as pd

df = pd.DataFrame(data=[[1,2,3,4],[4,5,6,7],[7,8,8,8],[10,11,12,14]], columns=['X','X','Y','Z'])
df = df.T
print(df)
print(df.groupby(df.index).sum())
