import numpy as np
import pandas as pd

# create XOR dataset
X = np.random.randn(300, 2)
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

df = pd.DataFrame(X, columns=['A','B'])
df1 = pd.DataFrame(y, columns=['C'])
df = df.join(df1['C'])

# recode True/False to 1/-1
df['C'] = df['C'].apply(lambda x: 1 if x == True else -1)
print(df)

df.to_csv('xor.csv', index=False)
