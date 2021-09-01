#date: 2021-09-01T13:24:25Z
#url: https://api.github.com/gists/38930b040f206e59845fbc30abf8d3df
#owner: https://api.github.com/users/orenmatar

sigma = prophet_obj.params["sigma_obs"][0]
historical_variance = np.random.normal(scale=sigma, size=sample_trends.shape)
full_samples = sample_trends + historical_variance