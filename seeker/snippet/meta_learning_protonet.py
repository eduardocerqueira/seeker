#date: 2021-09-28T17:07:35Z
#url: https://api.github.com/gists/0b0ab30a13231bcc7d8891ca7c20ae66
#owner: https://api.github.com/users/tchaton

# compute prototypes (c.f centroids) based on ground truth labels
centroids_shots = model_clone(task_shots)

# 1. compute distances to centroids
# 2. Assign to the closest centroid and assign it as its label
# 3. Compute `cross_entropy` using softmax over distances.
meta_loss += compute_assignement_to_centroids(task_queries, centroids_shots)