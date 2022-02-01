#date: 2022-02-01T16:58:16Z
#url: https://api.github.com/gists/8697b22228157fc67ba3eec790b5d85b
#owner: https://api.github.com/users/suhanacharya

from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
import sklearn.metrics as sm
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt

dataset = load_iris()
X = pd.DataFrame(dataset.data)
X.columns = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]
y = pd.DataFrame(dataset.target)
y.columns = ["Targets"]
plt.figure(figsize=(14, 7))
colormap = np.array(["red", "lime", "black"])

# REAL PLOT
plt.subplot(1, 3, 1)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y.Targets], s=40)
plt.title("Real")

# KMeans -PLOT
plt.subplot(1, 3, 2)
model = KMeans(n_clusters=3)
model.fit(X)
predY = np.choose(model.labels_, [0, 1, 2]).astype(np.int64)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[predY], s=40)
plt.title("KMeans")

# GMM PLOT
scaler = preprocessing.StandardScaler()
scaler.fit(X)
xsa = scaler.transform(X)
xs = pd.DataFrame(xsa, columns=X.columns)
gmm = GaussianMi    xture(n_components=3)
gmm.fit(xs)
y_cluster_gmm = gmm.predict(xs)
plt.subplot(1, 3, 3)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y_cluster_gmm], s=40)
plt.title("GMM Classification")