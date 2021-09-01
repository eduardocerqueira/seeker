#date: 2021-09-01T13:11:03Z
#url: https://api.github.com/gists/3dde45c268a5379d8fa42ab6838f97cb
#owner: https://api.github.com/users/orenmatar

future_t_time = ((forecast_df["ds"] - prophet_obj.start) / prophet_obj.t_scale).values
single_diff = np.diff(future_t_time).mean()  # single_diff is 1/len(train_data)
likelihood = len(prophet_obj.changepoints_t) * single_diff