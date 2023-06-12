#date: 2023-06-12T16:49:55Z
#url: https://api.github.com/gists/de3e5aab193126949dd663b59e471564
#owner: https://api.github.com/users/AbSsEnT

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
