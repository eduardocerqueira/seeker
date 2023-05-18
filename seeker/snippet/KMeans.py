#date: 2023-05-18T16:50:51Z
#url: https://api.github.com/gists/d4e3e0ad5de8d6c0aadb4b982c277eeb
#owner: https://api.github.com/users/yusufbrima

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import random 
# set the seed
random.seed(123)
np.random.seed(123)

# make blobs

X, y = ds.make_blobs(n_samples=200, centers=3, n_features=2, random_state=123, cluster_std=1.5, random_state=123)
K = 3

idx =  np.random.choice(X.shape[0], K, replace=False)
centroids = X[idx, :]

# plot the blobs
plt.figure(figsize=(8,6))
scatter  = plt.scatter(X[:,0], X[:,1],s=100)  # c = list(y)
plt.scatter(centroids[:,0], centroids[:,1], c = 'red', marker = 'D', s = 100)
# add text to the centroids
for i in range(K):
    plt.text(centroids[i,0], centroids[i,1], f"Centriod {str(i)}", fontsize=12)

plt.xlabel(r"$X_1$", fontsize=14)
plt.ylabel(r"$X_2$", fontsize=14)
plt.title("Before KMeans Clustering", fontsize=14, fontweight='bold' )
plt.savefig("KMeans_Before_Clustering.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))


def evaluate_point(point, centroids):
    distances = [euclidean_distance(point, centroid) for centroid in centroids]
    closest_centroid = np.argmin(distances)
    return closest_centroid

# Manual KMeans
def KMeans(X, K, max_iter = 100):
    c_idx =  np.random.choice(X.shape[0], K, replace=False)
    centroids = X[c_idx, :]
    for i in range(max_iter):
        # assign the points to the closest centroid
        clusters = [evaluate_point(point, centroids) for point in X]
        clusters = np.array(clusters)
        # update the centroids
        for i in range(K):
            idx = np.where(clusters == i)
            centroids[i] = X[idx , :].mean( axis = 1)
    return clusters, centroids
 

clusters, centroids = KMeans(X, K, max_iter = 100)


# plot the blobs
plt.figure(figsize=(8,6))
scatter  = plt.scatter(X[:,0], X[:,1],s=100, c = list(clusters))
plt.scatter(centroids[:,0], centroids[:,1], c = 'red', marker = 'D', s = 100)
# add text to the centroids
for i in range(K):
    plt.text(centroids[i,0], centroids[i,1], f"Centriod {str(i)}", fontsize=12)


plt.xlabel(r"$X_1$", fontsize=14)
plt.ylabel(r"$X_2$", fontsize=14)
plt.title("After KMeans Clustering", fontsize=14, fontweight='bold' )
plt.savefig("KMeans_After_Clustering.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()