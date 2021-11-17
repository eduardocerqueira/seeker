#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고 df.iloc

'''
sort_values: 컬럼 값으로 정렬
sort_index: 인덱스 값으로 정렬
'''

import seaborn as sns

df = sns.load_dataset("mpg")
# print(df)

# 1. 컬럼 값으로 정렬 df.sort_values(by 컬럼명), 기본은 오름차순
copy_df = df.sort_values(by="mpg")
print(copy_df.head(10))  # 데이터 프레임 반환
copy_df = df.sort_values(by="mpg", ascending=False)  # 내림차순
print(copy_df.head(10))  # 데이터 프레임 반환

# 2. 다중정렬 df.sort_values(by =[컬럼명1, 컬럼명2]
copy_df = df.sort_values(by=["mpg", "displacement"])
print(copy_df[["mpg", "displacement"]].head(20))  # 데이터 프레임 반환

# list_value = ["a", "bbb", "CCCC", "DDDDDD"]
# list_value.sort(key = len)
# print(list_value)

# 특정함수로 정렬(나중에)
# print("name 컬럼 문자열 길이로 정렬")
# copy_df = df.sort_values(by="name",
# print(copy_df.info())