#date: 2021-11-17T16:51:11Z
#url: https://api.github.com/gists/3452f645eb100eb60bd9b8033d18f1b7
#owner: https://api.github.com/users/chinshin5513

import numpy as np # 별칭 주기, 앞으로는 np. 으로 사용이 가능하다.

'''
병합(Merge)
1. 열 병합(np.hstack / np.column_stack / np.concatenate( axis=1) : 메모리가 누워져 있음
[ [x1 x2 x3] [x4]
  [y1 y2 y3] [y4]
  [z1 z2 z3] [z4] ]
  
2. 행 병합(np.vstack / np.row_stack / np.concatenate( axis=0) : 메모리가 세워져 있음
[ [x1 x2 x3]
  [y1 y2 y3]
  [z1 z2 z3]
  [x4 y4 z4] ]
'''

arr = np.arange(1, 10).reshape(3, 3)
print(arr)
arr2 = arr * 2
print(arr2)
# 병합이 되려면, 행, 렬이 같이야 함
# 1. 열병합(np.hstack / np.column_stack / np.concatenate( axis =1)
c_merge = np.hstack((arr, arr2))  # 튜플로 들어간다.
c_merge = np.column_stack((arr, arr2))  # 튜플로 들어간다.
c_merge = np.concatenate([arr, arr2], axis=1)  # 튜플, 리스트로 들어간다.
print(c_merge)  # 오른쪽 열에  arr2가 들어감

# 2. 행 병합(np.hstack / np.column_stack / np.concatenate( axis =1)
arr = np.arange(1, 10).reshape(3, 3)
print(arr)
arr2 = arr * 2
print(arr2)
r_merge = np.vstack((arr, arr2))  # 튜플로 들어간다.
print(r_merge)
r_merge = np.row_stack((arr, arr2))  # 튜플로 들어간다.
print(r_merge)
r_merge = np.concatenate([arr, arr2], axis=0)  # 튜플, 리스트로 들어간다
print(r_merge)  # 아래 행에 arr2가 들어감
