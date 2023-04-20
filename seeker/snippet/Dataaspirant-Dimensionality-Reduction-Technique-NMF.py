#date: 2023-04-20T17:05:28Z
#url: https://api.github.com/gists/45fbf15e037d160ba1ec78119a54d145
#owner: https://api.github.com/users/saimadhu-polamuri

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.datasets import load_digits

# Load digits dataset
digits = load_iris()
X = digits.data

# Perform NMF
nmf = NMF(n_components=2, init='random', random_state=0)
X_nmf = nmf.fit_transform(X)

# Visualize the results
plt.figure()
plt.grid()
plt.scatter(X_nmf[:, 0], X_nmf[:, 1], c=digits.target)
plt.colorbar()
plt.title('NMF of digits dataset')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()
