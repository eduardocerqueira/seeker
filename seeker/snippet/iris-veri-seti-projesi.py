#date: 2026-03-10T17:32:46Z
#url: https://api.github.com/gists/86180d5bb481cb81bf7dd392056cbd2b
#owner: https://api.github.com/users/devreyakan

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# csv dosyamızı okuyoruz.
dataset = pd.read_csv("iris.csv")