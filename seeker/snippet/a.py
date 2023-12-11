#date: 2023-12-11T16:58:49Z
#url: https://api.github.com/gists/627d5477989f311408de0d7ba9c280bc
#owner: https://api.github.com/users/abevieiramota

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Create a predictor X
X = np.random.default_rng().normal(size=100)
# Create a Noise Vector e
e = np.random.default_rng().normal(size=100)

B_0 = 0.5
B_1 = -0.3
B_2 = 3
B_3 = 0.7

Y = B_0 + B_1 * X + B_2 * (X ** 2) + B_3 * (X ** 3) + e

features = np.array([X**i for i in range(1, 11)]).T

best_rss_set = [None]
best_features_idx = []

features_idx = [i for i in range(features.shape[1])]

while features_idx:
    best_rss = None
    best_idx = None
    for feature_idx in features_idx:
        features_set = features[:, best_features_idx + [feature_idx]]
        model = LinearRegression().fit(features_set, Y)
        y_pred = model.predict(features_set)
        Rss = np.sum((Y - y_pred)**2)
        # Descobrir melhor RSS e melhor Índice
        if best_rss == None or Rss < best_rss:
            best_rss = Rss
            best_idx = feature_idx
    # Alimentar a best_rss_set e a best_features_set
    best_rss_set.append(best_rss)
    best_features_idx.append(best_idx)
    features_idx.remove(best_idx)
  
best_rss_set