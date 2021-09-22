#date: 2021-09-22T17:16:25Z
#url: https://api.github.com/gists/570b08d8052d71a94ee57b188abfbf90
#owner: https://api.github.com/users/sachinsdate

import math
import numpy as np
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
from scipy.stats import poisson
from scipy.stats import binom
from patsy import dmatrices
import statsmodels.graphics.tsaplots as tsa
from matplotlib import pyplot as plt

#Let's load the data set into memory using statsmodels
strikes_dataset = sm.datasets.get_rdataset(dataname='StrikeNb', package='Ecdat')

#Print out the data set
print(strikes_dataset.data)

#We'll consider the first 92 data points as the training set and the remaining
# 16 data points as the test data set
strikes_data = strikes_dataset.data.copy()
strikes_data_train = strikes_data.query('time<=92')
strikes_data_test = strikes_data.query('time>92').reset_index().drop('index', axis=1)

#Here is our regression expression. strikes is the dependent variable and
# output is our explanatory variable.
#The intercept of regression is assumed to be present
expr = 'strikes ~ output'

#We'll use Patsy to carve out the X and y matrices. Patsy will automatically
# add a regression intercept column to X
y_train, X_train = dmatrices(expr, strikes_data_train, return_type='dataframe')
print(y_train)
print(X_train)
y_test, X_test = dmatrices(expr, strikes_data_test, return_type='dataframe')
print(y_test)
print(X_test)


class PoissonINAR(GenericLikelihoodModel):
    def __init__(self, endog, exog, **kwds):
        super(PoissonINAR, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        #Fetch the parameters gamma and beta that we would be optimizing
        gamma = params[-1]
        beta = params[:-1]
        #Set y and X
        y = self.endog
        y = np.array(y)
        X = self.exog
        #Compute rho as a function of gamma
        rho = 1.0/(1.0+math.exp(-gamma))
        #Compute the Poisson mean mu as a dot product of X and Beta
        mu = np.exp(X.dot(beta))
        #Init the list of loglikelihhod values, one value for each y
        ll = []
        #Compute all the log-likelihood values for the Poisson INAR(1) model
        for t in range(len(y)-1,0,-1):
            prob_y_t = 0
            for j in range(int(min(y[t], y[t-1])+1)):
                prob_y_t += poisson.pmf((y[t]-j), mu[t]) * binom.pmf(j, y[t-1], rho)
            ll.append(math.log(prob_y_t))
        ll = np.array(ll)
        print('gamma='+str(gamma) + ' rho='+str(rho) + ' beta='+str(beta) + ' ll='+str(((-ll).sum(0))))
        #return the negated array of  log-likelihood values
        return -ll

    def fit(self, start_params=None, maxiter=1000, maxfun=5000, **kwds):
        #Add the gamma parameter to the list of exogneous variables that
        # the model will optimize
        self.exog_names.append('gamma')
        if start_params == None:
            #Start with some initial values of Beta and gamma
            start_params = np.append(np.ones(self.exog.shape[1]), 1.0)
        #Call super.fit() to start the training
        return super(PoissonINAR, self).fit(start_params=start_params,
            maxiter=maxiter, maxfun=maxfun, **kwds)

    def predict(self, params, exog=None, *args, **kwargs):
        #Fetch the optimized values of parameters gamma and beta
        fitted_gamma = params[-1]
        fitted_beta = params[:-1]
        X = np.array(exog)
        #Compute rho as a function of gamma
        rho = 1.0/(1.0+math.exp(-fitted_gamma))
        #Fetch the Intercept and the regression variables,
        # except for the last column which contains the lagged y values
        X = exog[:,:-1]
        #Fetch the lagged y values
        y_lag_1 = exog[:,-1]
        #Compute the predicted y using the Poisson INAR(1) model's equation
        y_pred = rho * y_lag_1 + np.exp(X.dot(fitted_beta))
        return y_pred


#Let's create an instance of the Poisson INAR(1) model class
inar_model = PoissonINAR(y_train, X_train)
inar_model_results = inar_model.fit()

#Print the model training summary
print(inar_model_results.summary())

#Prepare the X matrix for prediction
X_test['y_lag_1'] = y_test.shift(1)
X_test = X_test.fillna(0)

#Generate predictions on the test data set
inar_predictions = np.round(inar_model_results.predict(exog=X_test))

print(inar_predictions)

#plot the predicted counts versus the actual counts for the test data
predicted_counts=inar_predictions
actual_counts = y_test['strikes']
fig = plt.figure()
fig.suptitle('Predicted versus actual strike counts')
predicted, = plt.plot(X_test.index, predicted_counts, 'go-', label='Predicted counts')
actual, = plt.plot(X_test.index, actual_counts, 'ro-', label='Actual counts')
plt.legend(handles=[predicted, actual])
plt.show()
