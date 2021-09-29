#date: 2021-09-29T17:13:12Z
#url: https://api.github.com/gists/d6136a05b782f03ac01d2f548652eae0
#owner: https://api.github.com/users/erykml

s_size = 100000 # sample size

rv_std_norm = np.random.normal(size=s_size) # standard normal
rv_norm = np.random.normal(loc=1, scale=2.5, size=s_size) # normal
rv_skew_norm = scs.skewnorm.rvs(a=5, size=s_size) # skew normal