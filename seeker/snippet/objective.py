#date: 2022-01-05T16:55:57Z
#url: https://api.github.com/gists/da9b2602ae7f3cdba75bfdeb2e2dc0e2
#owner: https://api.github.com/users/yashprakash13

def objective(n_trials):
  params = {
        "n_estimators": n_trials.suggest_int("n_estimators", 100, 2000, step=100),
        "learning_rate": n_trials.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "max_depth": n_trials.suggest_int("max_depth", 3, 15),
        "n_iter_no_change": 50,
    }
  dtrain = xgb.DMatrix(data = X_train, label = y_train)
  dval = xgb.DMatrix(data = X_test, label = y_val)

  regressor = xgb.train(params, dtrain) 
  y_pred = regressor.predict(dval)
  rmse = mean_squared_error(y_val, y_pred, squared=False)

  return rmse