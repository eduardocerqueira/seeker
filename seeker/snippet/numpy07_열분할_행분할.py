#date: 2021-11-17T16:51:11Z
#url: https://api.github.com/gists/3452f645eb100eb60bd9b8033d18f1b7
#owner: https://api.github.com/users/chinshin5513

import numpy as np # 별칭 주기, 앞으로는 np. 으로 사용이 가능하다.

'''
분할(split)
1. 열분할(np.hsplit / np.split( axis=1) : 메모리가 누워져 있음
[ [x1 x2 x3]   /   [x4]
  [y1 y2 y3]   /   [y4]
  [z1 z2 z3]   /   [z4] ]
  
2. 행 분할(np.vsplit / np.split( axis=0) : 메모리가 세워져 있음
[ [x1 x2 x3]
  [y1 y2 y3]
  [z1 z2 z3]
  /  [x4 y4 z4] ]
'''

# 열 분할
arr = np.arange(1, 13).reshape(3,4)
print(arr)
c_split = np.hsplit(arr, 2)  # 분할하고자 하는 행렬의 숫자를 입력
print(c_split)  # 2개의 행렬로 균등하게 분할
c1, c2 = np.hsplit(arr, 2)
print(c1, c2)
c1,  c2 = np.split(arr, 2, axis=1)
print(c1, c2)

# 열 분할
arr = np.arange(1, 17).reshape(4, 4)
print(arr)
c_split = np.vsplit(arr, 2)  # 분할하고자 하는 행렬의 숫자를 입력
print(c_split)  # 2개의 행렬로 균등하게 분할
c1, c2 = np.vsplit(arr, 2)
print(c1, c2)
c1,  c2 = np.split(arr, 2, axis=0)
print(c1, c2)
