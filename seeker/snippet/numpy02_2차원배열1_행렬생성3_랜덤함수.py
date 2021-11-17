#date: 2021-11-17T16:51:11Z
#url: https://api.github.com/gists/3452f645eb100eb60bd9b8033d18f1b7
#owner: https://api.github.com/users/chinshin5513

import numpy as np # 별칭 주기, 앞으로는 np. 으로 사용이 가능하다.

'''

'''
# 1. 1차원으로 생성
arr = np.random.random(2)
print(arr)

# 2. 2차원으로 생성
arr = np.random.random((2,3))  # 튜플로 넣어줌
print(arr)

# 2. 2차원으로 생성
arr = np.random.rand(2, 3)  # 균등분포 (0~1)
print(arr)
arr = np.random.randn(2, 3)  # 정규분포
print(arr)
arr = np.random.randint(low=0, high=3, size=(2, 3))  # 다차원 범위 지정을 위한 튜플 사용
print(arr)
