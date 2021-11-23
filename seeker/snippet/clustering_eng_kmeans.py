#date: 2021-11-23T17:07:45Z
#url: https://api.github.com/gists/b6643035d0dce5f4c315da0951a67027
#owner: https://api.github.com/users/andrea-dagostino

from sklearn.cluster import KMeans

# initialize kmeans with 3 centroids
kmeans = KMeans(n_clusters=3, random_state=42)
# fit the model
kmeans.fit(X)
# store cluster labels in a variable
clusters = kmeans.labels_