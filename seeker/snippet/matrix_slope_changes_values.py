#date: 2021-09-01T13:19:08Z
#url: https://api.github.com/gists/fc2a0400567d369b108a624f3629f5f5
#owner: https://api.github.com/users/orenmatar

deltas = prophet_obj.params["delta"][0]. # deltas of slopes in the training data
mean_delta = np.mean(np.abs(deltas)) + 1e-8  # add epsilon, to avoid 0s
shift_values = np.random.laplace(0, mean_delta, size=bool_slope_change.shape)
matrix = shift_values * bool_slope_change