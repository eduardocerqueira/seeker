#date: 2023-11-08T17:05:36Z
#url: https://api.github.com/gists/a5296c217588c1628626bc1fd3d80a80
#owner: https://api.github.com/users/woldemarg

import numpy as np
from sklearn.metrics import pairwise_distances


def _calculate_dunn_index(data: np.ndarray,
                          labels: np.ndarray,
                          centroids: np.ndarray) -> float:

    # https://gist.github.com/douglasrizzo/cd7e792ff3a2dcaf27f6
    # https://github.com/jqmviegas/jqm_cvi/blob/master/jqmcvi/base.py
    # https://python.engineering/dunn-index-and-db-index-cluster-validity-indices-set/

    cluster_distances = []

    for cluster_label in np.unique(labels):

        cluster_points = data[labels == cluster_label]

        if len(cluster_points) > 1:
            intra_cluster_distances = pairwise_distances(
                cluster_points, metric='euclidean', n_jobs=-1)

            cluster_distances.append(np.mean(intra_cluster_distances))

    inter_cluster_distances = pairwise_distances(
        centroids, metric='euclidean', n_jobs=-1)

    min_inter_cluster_distance = np.min(
        inter_cluster_distances[inter_cluster_distances > 0])

    max_intra_cluster_distance = np.max(cluster_distances)

    dunn_index = min_inter_cluster_distance / max_intra_cluster_distance

    return dunn_index