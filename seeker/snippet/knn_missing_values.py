#date: 2022-09-12T17:09:06Z
#url: https://api.github.com/gists/5a3e88dee33b89d2d1a7668ee96db046
#owner: https://api.github.com/users/everydaycodings

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.metrics import accuracy_score

df = pd.read_csv('train.csv')

# n_neighbour is Number of neighboring samples to use for imputation you have to experiment with this number.
knn = KNNImputer(n_neighbors=3,weights='distance')

new_df = knn.fit_transform(df)