#date: 2022-10-28T17:11:43Z
#url: https://api.github.com/gists/ff6500339a896fd1b14ff923c9053843
#owner: https://api.github.com/users/rpromoditha

# Getting data
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
classes=cancer.target_names
X = cancer.data
y = cancer.target

# Importing PCA visualizer
from yellowbrick.features import PCA

# Creating the 2D scatter plot by utilizing PCA
visualizer = PCA(scale=True, projection=2, 
                 classes=classes)

visualizer.fit_transform(X, y)

# Saving plot in PNG format
visualizer.show(outpath="PC_Plot_2D.png")

