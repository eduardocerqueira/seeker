#date: 2021-09-10T17:03:42Z
#url: https://api.github.com/gists/05c52c9899c7e1437b8baf9cb641f28e
#owner: https://api.github.com/users/DanJRoot

# RFE Model
from sklearn.feature_selection import RFE

estimator = XGBRegressor()
selector = RFE(estimator, n_features_to_select=20, step=1)
selector.fit(X_train, y_train)
print(dict(zip(X_train.columns, selector.ranking_)))

X_train.columns[selector.support_]
selector.predict(X_test)

y_pred = selector.predict(X_valid)

mae = mean_absolute_error(y_pred, y_valid)
print(mae)
prediction = selector.predict(X_test)