#date: 2021-12-24T17:10:23Z
#url: https://api.github.com/gists/82f1f9cecfec3fc8fb5f343d7f9d3deb
#owner: https://api.github.com/users/ROHIT318

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
# X_train variable contains data.
x_min, x_max = X_train.min() - 1, X_train.max() + 1
y_min, y_max = X_train.min() - 1, X_train.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = knn_clf.predict(np.c_[xx.ravel(), yy.ravel()])
print(Z.shape)
# Put the result into a color plot
Z = Z.reshape(xx.shape)
print(yy.shape)
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, cmap=cmap_light)

sns.scatterplot(
        x = X_train[:, 0],
        y = X_train[:, 1],
        hue=dataset.target_names[y_label],
        palette=color_palette,
        alpha=1.0,
        edgecolor="blue",
    )