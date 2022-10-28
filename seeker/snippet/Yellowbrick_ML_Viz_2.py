#date: 2022-10-28T17:17:17Z
#url: https://api.github.com/gists/0c9ecd0c09a49969249e5af8ec35914e
#owner: https://api.github.com/users/rpromoditha

visualizer = PCA(scale=True, projection=3, 
                 classes=classes)

visualizer.fit_transform(X, y)
visualizer.show(outpath="PC_Plot_3D.png")

