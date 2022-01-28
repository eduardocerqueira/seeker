#date: 2022-01-28T17:02:40Z
#url: https://api.github.com/gists/162243ba5ba75545b333b379bf649ca8
#owner: https://api.github.com/users/sriram-pasupuleti

L2 = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
train_MAE = []
test_MAE = []
train_MAPE = []
test_MAPE = []

for l2 in L2:
  SGD = SGDRegressor(loss='squared_loss', penalty='l2', alpha=l2)
  RGD = RANSACRegressor(base_estimator = SGD)
  RGD.fit(X_train, y_train)
  train_pred = RGD.predict(X_train)
  test_pred = RGD.predict(X_test)
  train_MAE.append(mean_absolute_error(y_train,train_pred))
  test_MAE.append(mean_absolute_error(y_test, test_pred))
  train_MAPE.append(mean_absolute_percentage_error(y_train,train_pred))
  test_MAPE.append(mean_absolute_percentage_error(y_test,test_pred))