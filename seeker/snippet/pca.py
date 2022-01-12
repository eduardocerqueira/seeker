#date: 2022-01-12T17:06:18Z
#url: https://api.github.com/gists/0aee7ecca1746f7dac5f5ba30a7231d7
#owner: https://api.github.com/users/haykaza

# run pca
pca = PCA(n_components = 127)
output_pca = pca.fit_transform(X.to_numpy())

print("original shape:   ", X.shape)
print("transformed shape:", output_pca.shape)