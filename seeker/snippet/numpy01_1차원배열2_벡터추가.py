#date: 2021-11-17T16:51:11Z
#url: https://api.github.com/gists/3452f645eb100eb60bd9b8033d18f1b7
#owner: https://api.github.com/users/chinshin5513

import numpy as np

v1 = np.array([9, 8, 7, 6, 5])

# 1. 추가 ==> np.append(arr, 값)
v1_copy = np.append(v1, 100)
print(v1) # [9 8 7 6 5]
print(v1_copy) # [  9   8   7   6   5 100]

# 2. 삽입 ==> np.insert
v1_copy = np.insert(v1, 0, 100)
print(v1_copy) # [100   9   8   7   6   5]

# numpy에서 색인은 여러개 지정 가능:  [idx1, idx2] ==> fancy 색인이라고 부른다.
v1_copy = np.insert(v1, [0,2], 100) # [100   9   8 100   7   6   5]
print(v1_copy) #
v1_copy = np.insert(v1, [1,2,3], 100) # [100   9   8 100   7   6   5]
print(v1_copy) #
#v1_copy = np.insert(v1, slice(1,4), 100) # [start:end:step] 지원안됨
#print(v1_copy) #