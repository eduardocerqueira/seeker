#date: 2026-03-10T17:39:34Z
#url: https://api.github.com/gists/98facc05ad48ea4660edb5213975fde4
#owner: https://api.github.com/users/devreyakan

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Verilerimizi okuyoruz.
dataset = pd.read_csv("housing.csv",delim_whitespace=True)

# Bağımlı değişkenleri ve bağımsız değişkenleri ayırıyoruz.
X = dataset.iloc[:,0:13]
Y = dataset.iloc[:,13]