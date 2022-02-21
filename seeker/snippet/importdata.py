#date: 2022-02-21T17:02:40Z
#url: https://api.github.com/gists/31edc510d3d98bb7aa5b78062d79d4e5
#owner: https://api.github.com/users/nxbisgin

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, auc, roc_curve, confusion_matrix
from sklearn.inspection import permutation_importance

heart_df = pd.read_csv('./data/heart.csv')
heart_df.head()