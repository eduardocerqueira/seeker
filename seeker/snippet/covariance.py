#date: 2024-01-17T16:47:57Z
#url: https://api.github.com/gists/1a033e6a519c2eb1a2d3ddbe9f6ecfd1
#owner: https://api.github.com/users/pratapsdev11

def covariance(x):
 cov_mat=(standardize_data(x).T @ standardize_data(x))/(standardize_data(x).shape[0]-1)
 return cov_mat