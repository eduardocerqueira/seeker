#date: 2022-04-20T17:21:50Z
#url: https://api.github.com/gists/71b4db768c7aa6bd306a299192d32896
#owner: https://api.github.com/users/FerusAndBeyond

import pandas as pd

# list of dicts to pd.DataFrame
df = pd.DataFrame([
    { "a": 5, "b": 6 },
    { "a": 6, "b": 7 }
])
# df =
#    a  b
# 0  5  6
# 1  6  7

# a dict to pd.Series
srs = pd.Series({ "a": 5, "b": 6 })
# srs =
# a    5
# b    6
# dtype: int64