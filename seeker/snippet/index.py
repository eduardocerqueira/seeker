#date: 2022-01-24T17:12:00Z
#url: https://api.github.com/gists/03e503f87a8302adb252a8a4084c5b7b
#owner: https://api.github.com/users/chinph

pd.crosstab(index = df.pickup_borough, columns = df.payment,\
                  values = df.total, aggfunc = 'sum',\
                  normalize = "index")