import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, 1:4].values
y = dataset.iloc[:, 0].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [2])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

dataset_test = pd.read_csv('private_test_x.csv')
X_test = dataset_test.iloc[:, 1:4].values

X_test[:, 2] = labelencoder_X_1.transform(X_test[:, 2])
X_test = onehotencoder.transform(X_test).toarray()
X_test = X_test[:, 1:]

y_expected = dataset_test.iloc[:, 0].values
y_pred = regressor.predict(X_test)
y_pred = y_pred.astype(np.int64)

xx = dataset_test.iloc[:, 1:4].values
xx = xx.tolist()
#xx[:, 0:2] = xx[:, 0:2].astype(np.int64)
output = np.column_stack((y_pred, xx))
headers = ["Usage", "Timestep", "InventoryCode", "Domain"]
op = np.row_stack((headers, output))
df = pd.DataFrame(op)
df.to_csv("privatetest.csv", index = False, header = None)

final = pd.read_csv("privatetest.csv")