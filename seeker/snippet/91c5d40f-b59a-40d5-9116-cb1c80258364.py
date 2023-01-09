#date: 2023-01-09T16:52:55Z
#url: https://api.github.com/gists/03850922adc38e8f8ac819d61fe50e23
#owner: https://api.github.com/users/christopherDT

from sklearn.metrics import r2_score

# forecast accuracy
future.index = future.ds
future['y'] = df_test['y'].values

print(f'R-squared = {round(r2_score(future["y"], future["yhat"]), 2)}')
print(f'mean squared error = {round(mean_squared_error(future["y"], future["yhat"]),2)}')

future[['y', 'yhat']].plot(figsize=(8,5))
