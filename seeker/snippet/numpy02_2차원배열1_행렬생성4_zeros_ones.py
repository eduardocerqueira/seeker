#date: 2021-11-17T16:51:11Z
#url: https://api.github.com/gists/3452f645eb100eb60bd9b8033d18f1b7
#owner: https://api.github.com/users/chinshin5513

import numpy as np # 별칭 주기, 앞으로는 np. 으로 사용이 가능하다.

'''

'''
# 1. 영행렬
arr = np.zeros((2, 3))
print(arr)

arr = np.zeros((2, 3), dtype=int)
print(arr)

# 2. 1행렬
arr = np.ones((2, 3), dtype=int)
print(arr)

# 3. empty
arr = np.empty((2, 3), dtype=int)
print(arr)  # 랜덤하게 숫자 출력

# 4. full
arr = np.full((2, 3), 100)
print(arr)  # 지정된 숫자 출력

# 5. 단위행렬
arr = np.eye((3), dtype=int)
print(arr)  # I 대각선이 1이고 나머지는 0인 행렬
