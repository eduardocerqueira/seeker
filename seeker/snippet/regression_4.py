#date: 2025-01-03T17:08:42Z
#url: https://api.github.com/gists/5a81134f3a2ff0146cdc1d7e3ae1f613
#owner: https://api.github.com/users/PieroPaialungaAI

from sklearn.decomposition import PCA
#Fitting the PCA model
pca = PCA(n_components=2)
#Creating the full dataset
X = np.hstack((X_1,X_2))
#Fitting the PCA
pca.fit(X)
#Transforming the dataset
X_PCA = pca.transform(X)