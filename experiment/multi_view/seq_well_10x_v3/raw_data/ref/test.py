import pandas as pd

sm_1 = pd.read_csv('sm_1_1.csv', index_col=0)
sm2 = pd.read_csv('sm_1_2.csv', index_col=0)

print(sm_1.to_numpy().sum())
print(sm2.to_numpy().sum())