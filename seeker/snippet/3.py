#date: 2022-03-09T17:13:41Z
#url: https://api.github.com/gists/325ef5847573f8a364021088a6489f2a
#owner: https://api.github.com/users/bengchew-lab

# create custom transformer
class DropFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        X_dropped = X.drop(self.variables, axis = 1)
        return X_dropped
      
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        return X.loc[:,self.variables]
      
class SimpleImputerCustom(BaseEstimator, TransformerMixin):
    def __init__(self, variables, strategy):
        self.variables = variables
        self.strategy = strategy
        self.imp = SimpleImputer(missing_values=np.nan,   
                    strategy=self.strategy)
    def fit(self, X, y = None):
        X_ = X.loc[:,self.variables]
        self.imp.fit(X_)
        return self
    def transform(self, X):
        X_ = X.loc[:,self.variables]
        X_transformed = pd.DataFrame(self.imp.transform(X_), 
                         columns= self.variables)
        X.drop(self.variables, axis= 1, inplace=True)
        X[self.variables] = X_transformed[self.variables].values
        return X
      
class DomainNumFE(BaseEstimator, TransformerMixin):
    def __init__(self, variables = None):
        self.variables = variables
    def fit(self, X, y =None):
        return self
    def transform(self, X):
        # source: https://www.kaggle.com/lavanyashukla01/how-i-made-top-0-3-on-a-kaggle-competition#Feature-Engineering
        X_ = X.copy()
        X_['HasWoodDeck'] = (X_['WoodDeckSF'] == 0) * 1
        X_['HasOpenPorch'] = (X_['OpenPorchSF'] == 0) * 1
        X_['HasEnclosedPorch'] = (X_['EnclosedPorch'] == 0) * 1     
        X_['Has3SsnPorch'] = (X_['3SsnPorch'] == 0) * 1
        X_['HasScreenPorch'] = (X_['ScreenPorch'] == 0) * 1
        X_['YearsSinceRemodel'] = X_['YrSold'].astype(int) - 
            X_['YearRemodAdd'].astype(int)
        X_['Total_Home_Quality'] = X_['OverallQual'] + 
            X_['OverallCond']
        X_['TotalSF'] = X_['TotalBsmtSF'] + X_['1stFlrSF'] + 
            X_['2ndFlrSF']
        X_['YrBltAndRemod'] = X_['YearBuilt'] + X_['YearRemodAdd']
        X_['Total_sqr_footage'] = (X_['BsmtFinSF1'] +   
            X_['BsmtFinSF2'] + X_['1stFlrSF'] + X_['2ndFlrSF'])
        X_['Total_porch_sf'] = (X_['OpenPorchSF'] + X_['3SsnPorch'] 
            + X_['EnclosedPorch'] + X_['ScreenPorch'] + 
            X_['WoodDeckSF'])
        X_['TotalBsmtSF'] = X_['TotalBsmtSF'].apply(lambda x: 
            np.exp(6) if x <= 0.0 else x)
        X_['2ndFlrSF'] = X_['2ndFlrSF'].apply(lambda x: np.exp(6.5) 
            if x <= 0.0 else x)
        X_['GarageArea'] = X_['GarageArea'].apply(lambda x: 
            np.exp(6) if x <= 0.0 else x)
        X_['LotFrontage'] = X_['LotFrontage'].apply(lambda x: 
            np.exp(4.2) if x <= 0.0 else x)
        X_['MasVnrArea'] = X_['MasVnrArea'].apply(lambda x: 
            np.exp(4) if x <= 0.0 else x)
        X_['BsmtFinSF1'] = X_['BsmtFinSF1'].apply(lambda x: 
            np.exp(6.5) if x <= 0.0 else x)
        X_['haspool'] = X_['PoolArea'].apply(lambda x: 1 if x > 0 
            else 0)
        X_['has2ndfloor'] = X_['2ndFlrSF'].apply(lambda x: 1 
            if x > 0 else 0)
        X_['hasgarage'] = X_['GarageArea'].apply(lambda x: 1 
            if x > 0 else 0)
        X_['hasbsmt'] = X_['TotalBsmtSF'].apply(lambda x: 1 
            if x > 0 else 0)
        return X_
      
class OneHotEncodercustom(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables
        self.ohe = OneHotEncoder(drop='first', 
            handle_unknown = 'ignore')
    def fit(self, X, y = None):
        X_ = X.loc[:,self.variables]
        self.ohe.fit(X_)
        return self
    def transform(self, X):
        X_ = X.loc[:,self.variables]
        X_transformed =   
            pd.DataFrame(self.ohe.transform(X_).toarray(), 
            columns= self.ohe.get_feature_names_out())
        X.drop(self.variables, axis= 1, inplace=True)
        X[self.ohe.get_feature_names_out()] = 
            X_transformed[self.ohe.get_feature_names_out()].values
    return X
  
class DomainCatFE(BaseEstimator, TransformerMixin):
    def __init__(self, variables = None):
        self.variables = variables
    def fit(self, X, y =None):
        return self
    def transform(self, X):
    # source: https://www.kaggle.com/lavanyashukla01/how-i-made-top-0-3-on-a-kaggle-competition#Feature-Engineering
    X_ = X.copy()
    X_['BsmtFinType1_Unf'] = 1*(X_['BsmtFinType1'] == 'Unf')
    X_['Total_Bathrooms'] = (X_['FullBath'] + (0.5 * X_['HalfBath'])  
        + X_['BsmtFullBath'] + (0.5 * X_['BsmtHalfBath']))
    X_['GarageCars'] = X_['GarageCars'].apply(lambda x: 0 
        if x <= 0.0 else x)
    X_['hasfireplace'] = X_['Fireplaces'].apply(lambda x: 1 
        if x > 0 else 0)
   return X_