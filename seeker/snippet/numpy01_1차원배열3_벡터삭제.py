#date: 2021-11-17T16:51:11Z
#url: https://api.github.com/gists/3452f645eb100eb60bd9b8033d18f1b7
#owner: https://api.github.com/users/chinshin5513

import numpy as np # 별칭 주기, 앞으로는 np. 으로 사용이 가능하다.

v1 = np.array([9, 8, 7, 6, 5])

# 1. 삭제 → np.delete(arr, idx, axis = 0) 위치로 삭제
v1_copy = np.delete(v1, 0)
print(v1_copy)  # [8 7 6 5]

v1_copy = np.delete(v1, [0, 1, 2])
print(v1_copy)  # [6 5]

v1 = np.array([9, 8, 7, 6, 5])
# 1. 삭제 → np.delete(arr, idx, axis = 0) 위치로 삭제
print(np.where(v1 == 5))  # 요소의 위치값을 반환해줌
v1_copy = np.delete(v1, np.where(v1 == 5))  # 요소의 위치값으로 구해서 삭제해야 함
print(v1_copy)  # [9 8 7 6]

