#date: 2023-07-26T16:36:45Z
#url: https://api.github.com/gists/d630c99e14ae2eee98371d0d37915dbd
#owner: https://api.github.com/users/viyaleta

import pandas as pd
from scipy.io import loadmat
from sklearn.mixture import GaussianMixture

# upload the datafile 
# http://odds.cs.stonybrook.edu/wine-dataset/
# Saket Sathe and Charu C. Aggarwal. LODES: Local Density meets Spectral Outlier Detection. SIAM Conference on Data Mining, 2016.
wine = loadmat('wine.mat')

# grab the features and create a dataframe
columns = ["alcohol", "malicacid", "ash", "alcalinity_of_ash", "magnesium",
          "total_phenols", "flavanoids", "nonflanoid_phenols", "proanthocyanins",
          "color_intensity", "hue", "0D280_0D315_of_diluted_wines", "proline"]

print('Total features:', len(columns))

df = pd.DataFrame(wine["X"], columns=columns)

# anomaly targets
y_true = wine["y"].flatten()

# fit the model with 2 components
gmm = GaussianMixture(n_components=2)
gmm.fit(df)

# get predicted data
component = gmm.predict(df)
proba = gmm.predict_proba(df)