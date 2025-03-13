import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pickle as pkl

iris = load_iris()

data = pd.DataFrame(iris.data, columns = iris.feature_names)
data['target'] = iris.target

data.head()
x = data.iloc[:,:-1].values
y = data.iloc[:, -1].values


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 2)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

model = RandomForestRegressor()
model.fit(x_train, y_train)

# with open("iris_model.pkl", 'wb') as file:
#     pkl.dump(model, file)
# print(iris.target_names)

