#date: 2022-09-01T16:57:23Z
#url: https://api.github.com/gists/b7e9857e1c395d98b200fe9e9f5e63d6
#owner: https://api.github.com/users/MatheusHam

import warnings
import pandas as pd
from sklearn.datasets import make_classification

warnings.filterwarnings('ignore')

X, y = make_classification(
    n_samples=5000,
    n_features=30,
    n_redundant=15,
    n_clusters_per_class=1,
    weights=[0.50],
    class_sep=2,
    random_state=42
)

cols = []
for i in range(len(X[0])):
   cols.append(f"feat_{i}")
X = pd.DataFrame(X, columns=cols)
y = pd.DataFrame({"y": y})