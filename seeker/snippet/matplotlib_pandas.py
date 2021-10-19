#date: 2021-10-19T17:00:10Z
#url: https://api.github.com/gists/1adbd4cc19f2e84464f355dfc88feba7
#owner: https://api.github.com/users/hankyojeong

import pandas as pd

data_frame = pd.read_csv('datafile.csv')
data_frame.plot()

# 데이터를 누적값으로 바꿔줌
data_sum = data_frame.cumsum()
data_sum.plot()