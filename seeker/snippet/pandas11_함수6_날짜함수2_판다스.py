#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고
import seaborn as sns

# 2. 판다스 날짜함수 정리

# 1. 특정 날짜(str) datetime 변경
print("1. str → 날짜타입")
target_date = pd.to_datetime('2002/01/01')
target_date = pd.to_datetime('2002 02 03')
target_date = pd.to_datetime('2002-02-03')
# target_date = pd.to_datetime('2002,03,01')  # 인식하지 못함
print(target_date, type(target_date))  # 2002-01-01 00:00:00 <class 'pandas._libs.tslibs.timestamps.Timestamp'>

target_date = pd.to_datetime('2002:02:03', format = '%Y:%m:%d')  # 포맷 지정으로 가능
target_date = pd.to_datetime('2012년 02월 03일', format='%Y년 %m월 %d일')  # 시간이라서 연월일이 안 됨
target_date = pd.to_datetime('2002,03,01', format='%Y,%m,%d')  # 인식하지 못함
print(target_date, type(target_date))  #

df = pd.DataFrame({'year': [2015, 2016],
                        'month': [2, 3],
                        'day': [4, 5]})  # key값이 정해져 있음
copy_df = pd.to_datetime(df)
print(copy_df)

# 2.년-월-일 시:분:초
target_date = pd.to_datetime("2002년05월05일 12시12분13초", format="%Y년%m월%d일 %H시%M분%S초")
print(target_date)  # 2002-05-05 12:12:13
