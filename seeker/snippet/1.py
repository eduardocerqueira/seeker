#date: 2022-03-09T17:11:36Z
#url: https://api.github.com/gists/5fdf2a363340e01c7c4c3adfdc418beb
#owner: https://api.github.com/users/bengchew-lab

from sklearn.base import BaseEstimator, TransformerMixin

class DropFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        X_dropped = X.drop(self.variables, axis = 1)
        self.columns = X_dropped.columns
        return X_dropped