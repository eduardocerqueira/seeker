#date: 2022-01-11T17:11:05Z
#url: https://api.github.com/gists/f51247306ed42aa365cb5ccb194fd02a
#owner: https://api.github.com/users/thesfinox

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as sm

sns.set_theme()

#######################
#                     #
# ENTER THE DATA      #
#                     #
#######################

x_values = [...]
y_values = [...]

data = pd.DataFrame({'x': np.array(x_values), 'y': np.array(y_values)})

#######################
#                     #
# FIT THE DATA        #
#                     #
#######################

# linear w/ intercept
results = sm.ols('y ~ x', data).fit()

# linear w/o intercept
# results = sm.ols('y ~ x -1', data).fit()

# quadratic w/intercept
# results = sm.ols('y ~ x + np.power(x,2)', data).fit()

# quadratic w/o intercept
# results = sm.ols('y ~ x + np.power(x,2) -1', data).fit()

# linear w/ intercept
print(results.summary(yname='y_values',
                      xname=['intercept', 'ang. coeff.'],
                      title='Regression w/ intercept'
                      )
      )  

# linear w/o intercept
# print(results.summary(yname='y_values',
#                       xname=['ang. coeff.'],
#                       title='Regression w/o intercept'
#                       )
#       ) 

# quadratic w/ intercept
# print(results.summary(yname='y_values',
#                       xname=['intercept', 'ang. coeff.', 'quad. coeff.'],
#                       title='Regression w/ intercept'
#                       )
#       )  

# quadratic w/o intercept
# print(results.summary(yname='y_values',
#                       xname=['ang. coeff.', 'quad. coeff.'],
#                       title='Regression w/o intercept'
#                       )
#       )  

#######################
#                     #
# PLOT THE DATA       #
#                     #
#######################

# linear w/ intercept
b, a = results.params.values
regression = lambda t: a * t + b

# linear w/o intercept
# a = results.params.values
# regression = lambda t: a * t

# quadratic w/ intercept
# c, b, a = results.params.values
# regression = lambda t: a * t * t + b * t + c

# quadratic w/o intercept
# b, a = results.params.values
# regression = lambda t: a * t * t + b * t

fig, ax = plt.subplots(figsize=(6, 5))

x = np.linspace(min(x_values), max(x_values), num=100)
y = regression(x)

sns.scatterplot(x=data['x'],
                y=data['y'],
                color='tab:blue',
                label='experimental',
                ax=ax
                )
                
sns.lineplot(x=x,
             y=y,
             linestyle='--',
             color='tab:red',
             label='best fit',
             ax=ax
             )

ax.set(xlabel='x-data', ylabel='y-data')

plt.tight_layout()
# plt.savefig('best_fit.pdf', dpi=300)
plt.show()
