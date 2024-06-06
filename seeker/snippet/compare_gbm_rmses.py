#date: 2024-06-06T16:54:37Z
#url: https://api.github.com/gists/d4d516e0770fb4a0af84731f72e7b4ef
#owner: https://api.github.com/users/zyuzuguldu

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

custom_gbm = CustomGradientBoostingRegressor(
    n_estimators=20, 
    learning_rate=0.1, 
    max_depth=1
)
custom_gbm.fit(x, y)
custom_gbm_rmse = mean_squared_error(y, custom_gbm.predict(x), squared=False)
print(f"Custom GBM RMSE:{custom_gbm_rmse:.15f}")

sklearn_gbm = GradientBoostingRegressor(
    n_estimators=20, 
    learning_rate=0.1, 
    max_depth=1
)
sklearn_gbm.fit(x, y)
sklearn_gbm_rmse = mean_squared_error(y, sklearn_gbm.predict(x), squared=False)
print(f"Scikit-learn GBM RMSE:{sklearn_gbm_rmse:.15f}")