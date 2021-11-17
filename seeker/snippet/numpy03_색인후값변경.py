#date: 2021-11-17T16:51:11Z
#url: https://api.github.com/gists/3452f645eb100eb60bd9b8033d18f1b7
#owner: https://api.github.com/users/chinshin5513

import numpy as np # 별칭 주기, 앞으로는 np. 으로 사용이 가능하다.
'''
 1차원 배열의 색인 및 값 변경
'''
arr = np.arange(10)
print(arr)  # [0 1 2 3 4 5 6 7 8 9]
arr[0] = 100  # 인덱싱 후 값 변경
print(arr)  # [100   1   2   3   4   5   6   7   8   9]
arr[:4] = 200  # 슬라이싱 후 값 변경
print(arr)  # [200 200 200 200   4   5   6   7   8   9]
arr[[0, 5]] = 300  # 팬시 색인 후 값 변경
print(arr)  # [300 200 200 200   4 300   6   7   8   9]
arr[arr < 100] = 999  # 불린으로 값 변경 가능
print(arr)  # [300 200 200 200 999 300 999 999 999 999]
