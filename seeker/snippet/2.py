#date: 2022-03-09T17:12:27Z
#url: https://api.github.com/gists/e0bac515e5856bd384905869b7480efc
#owner: https://api.github.com/users/bengchew-lab

class OneHotEncodercustom(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables
        self.ohe = OneHotEncoder(drop='first', handle_unknown = 'ignore')
  
    def fit(self, X, y = None):
        X_ = X.loc[:,self.variables]
        self.ohe.fit(X_)
        return self
      
    def transform(self, X):
        X_ = X.loc[:,self.variables]
        # get one-hot encoded feature in df format
        X_transformed = pd.DataFrame(self.ohe.transform(X_).toarray(), columns= self.ohe.get_feature_names_out())
        
        # Remove columns that are one hot encoded in original df
        X.drop(self.variables, axis= 1, inplace=True)
        
        # Add one hot encoded feature to original df
        X[self.ohe.get_feature_names_out()] = X_transformed[self.ohe.get_feature_names_out()].values
        return X