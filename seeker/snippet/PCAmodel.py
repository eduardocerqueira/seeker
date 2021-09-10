#date: 2021-09-10T17:08:38Z
#url: https://api.github.com/gists/c134d49ef6e46e4176f649144fd9f637
#owner: https://api.github.com/users/DanJRoot

from sklearn.decomposition import PCA

pca = PCA(n_components=5)
X_pca = X.select_dtypes(include=['float64', 'int'])
X_pca.fillna(value=0, inplace=True)
y_pca = y

pca_model = XGBRegressor(n_estimators=200, learning_rate=0.05)
pca_model.fit(X_train_pca, y_train_pca)
y_pred_pca = pca_model.predict(X_valid_pca)
mae_pca = mean_absolute_error(y_pred_pca, y_valid_pca)
print(mae_pca)
prediction_pca = pca_model.predict(X_test_pca)