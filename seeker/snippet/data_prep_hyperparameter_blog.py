#date: 2022-03-18T17:05:43Z
#url: https://api.github.com/gists/2b68ffb3a49f347f0b626486c3506c18
#owner: https://api.github.com/users/kyleziegler

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
import numpy as np

d = load_wine()
X, y = d.data, d.target

# X = X[:, np.newaxis, 2]
# X = np.array(X.iloc[:, 0]).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)

print(X_train[:1])
print(y_train[:1])
print(len(X_train))