#date: 2022-05-10T16:59:03Z
#url: https://api.github.com/gists/c0737d74c093a3b251f8ef6426a91133
#owner: https://api.github.com/users/erdogant

# Load libraries
from sklearn import manifold, decomposition

# Import library
import flameplot as flameplot

# Load mnist example data
X, y = flameplot.import_example()

# PCA: 50 PCs
mapX = decomposition.TruncatedSVD(n_components=50).fit_transform(X)

# tSNE: 2D
mapY = manifold.TSNE(n_components=2, init='pca').fit_transform(X)

# Quantify PCA(50) vs. tSNE
scores = flameplot.compare(mapX, mapY, n_steps=5)

# Plot
flameplot.plot(scores, xlabel='PCA (50d)', ylabel='tSNE (2d)')
