#date: 2023-06-30T17:07:46Z
#url: https://api.github.com/gists/8fda8e66d5594078d375ab4fd468d3cf
#owner: https://api.github.com/users/davidemastricci

metric = 'mape'
n_select = 1
best_model = experiment.compare_models(sort=metric,  n_select = n_select)
