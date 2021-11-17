#date: 2021-11-17T16:51:11Z
#url: https://api.github.com/gists/3452f645eb100eb60bd9b8033d18f1b7
#owner: https://api.github.com/users/chinshin5513

import numpy as np # 별칭 주기, 앞으로는 np. 으로 사용이 가능하다.

'''

'''
# 1. arange
arr = np.arange(15)  # 1차원 행렬
print(arr)

arr = np.arange(15).reshape(15, 1)  # 2차원 행렬로 변환
print(arr)

