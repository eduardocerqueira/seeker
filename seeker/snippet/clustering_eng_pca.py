#date: 2021-11-23T17:15:10Z
#url: https://api.github.com/gists/5e313f1ed63e85eb79f88b47cc2ce842
#owner: https://api.github.com/users/andrea-dagostino

from sklearn.decomposition import PCA

# initialize PCA with 2 components
pca = PCA(n_components=2, random_state=42)
# pass our X to the pca and store the reduced vectors into pca_vecs
pca_vecs = pca.fit_transform(X.toarray())
# save our two dimensions into x0 and x1
x0 = pca_vecs[:, 0]
x1 = pca_vecs[:, 1]