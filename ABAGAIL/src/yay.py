from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


df = pd.read_csv("dataset.csv", sep = ",", quotechar = '"')
df = pd.get_dummies(df)
X = df.ix[:, df.columns != 'Result']
Y = df['Result']


scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
allData = np.column_stack((X, Y))
np.savetxt("dataset_norm.csv", allData, delimiter=",")
