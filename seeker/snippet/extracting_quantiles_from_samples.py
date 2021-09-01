#date: 2021-09-01T13:25:02Z
#url: https://api.github.com/gists/42cf00a0eca2bb0f20713601039e1028
#owner: https://api.github.com/users/orenmatar

quantiles = np.array([10, 90])
quantiles = np.percentile(full_samples, quantiles, axis=0)
forecast_df["yhat_lower"] = forecast_df.yhat + quantiles[0]
forecast_df["yhat_upper"] = forecast_df.yhat + quantiles[1]