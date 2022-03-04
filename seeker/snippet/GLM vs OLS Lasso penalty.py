#date: 2022-03-04T17:02:06Z
#url: https://api.github.com/gists/08dbd72695e6d263546c973a9a5a330b
#owner: https://api.github.com/users/xiaowei1234

import numpy as np
from sklearn.linear_model import PoissonRegressor, Lasso

X_array = np.asarray([[1, 2], [1, 3], [1, 4], [1, 3]])
y = np.asarray([2, 2, 3, 2])
Preg_alpha_1 = PoissonRegressor(alpha=1., fit_intercept=False).fit(X_array, y)
print('alpha 1', Preg_alpha_1.coef_)
Preg_alpha_2 = PoissonRegressor(alpha=2., fit_intercept=False).fit(X_array/2., y)
print('alpha 2', Preg_alpha_2.coef_)
Lreg_alpha_1 = Lasso(alpha=1., fit_intercept=False).fit(X_array, y)
print('alpha 1 Lasso OLS', Lreg_alpha_1.coef_)
Lreg_alpha_2 = Lasso(alpha=2., fit_intercept=False).fit(X_array/2., y)
print('alpha 2 Lasso OLS', Lreg_alpha_2.coef_)