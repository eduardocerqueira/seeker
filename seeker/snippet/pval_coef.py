#date: 2022-01-12T17:19:25Z
#url: https://api.github.com/gists/f26b80354fad5b8ea211549da19ae276
#owner: https://api.github.com/users/haykaza

#normalize the data
X_norm = (X_Finbert - np.mean(X_Finbert)) / np.std(X_Finbert, 0)

#fit logistic regression model with normalized data
model = lr.fit(X_norm, y)

#create a dataframe consisting of the list of coefficients
coefs = pd.DataFrame(index = X_Finbert.columns, data = model.coef_[0], columns = ['Coefficients'])