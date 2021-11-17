#date: 2021-11-17T16:51:11Z
#url: https://api.github.com/gists/3452f645eb100eb60bd9b8033d18f1b7
#owner: https://api.github.com/users/chinshin5513

import numpy as np # 별칭 주기, 앞으로는 np. 으로 사용이 가능하다.
'''
 2차원 배열의 색인
  1. 파이썬 문법과 동일하게 사용
'''
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("4. arr[행, 열: 팬시 색인]", arr[0, [0, 2]])  # 4. arr[행, 팬시 색인] [1 3], 팬시 색인에는 대괄호가 필수다!
print("6. arr[행: 팬시 색인, 열]", arr[[0, 2], 1])  # 6. arr[행: 팬시 색인, 열] [2 8]
print("8. arr[행: fancy, 열: fancy]", arr[[0, 2], [0, 2]])  # 8. arr[행: fancy, 열: fancy] [1 9] 0행0열 + 2행2열

# 불린 색인 또한 가능



