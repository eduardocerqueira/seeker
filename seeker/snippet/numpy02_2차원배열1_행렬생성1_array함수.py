#date: 2021-11-17T16:51:11Z
#url: https://api.github.com/gists/3452f645eb100eb60bd9b8033d18f1b7
#owner: https://api.github.com/users/chinshin5513

import numpy as np # 별칭 주기, 앞으로는 np. 으로 사용이 가능하다.

'''
대괄호의 갯수에 따라 1차원, 2차원, 3차원으로 구분됨
1차원 : [    ]
2차원 : [[    ], [    ]]
3차원 : [[[  ],[  ],[  ]],[[  ],[  ],[  ]]]
'''

# 1. 직접 때려박기
list_value = [[10, 20, 30], [1, 2, 3]]
arr1 = np.array(list_value)
print(arr1, type(arr1))

list_value = [[[10, 20, 30], [1, 2, 3]], [[10, 20, 30], [1, 2, 3]]]
arr1 = np.array(list_value)
print(arr1, type(arr1))

# 2. 1차원을 2차원으로 변환하기
list_value = [1, 2, 3, 4, 5, 6]
'''
요소가 6개이므로, 1×6, 2×3, 3×2, 6×1 행렬이 만들어질 수 있다.
이 때 쓰는 명령어는 np.shape 함수
'''
arr1 = np.array(list_value)
print(arr1)
arr1.shape = (1, 6)  # 1×6, 튜플
print(arr1)
arr1.shape = (2, 3)  # 2×3
print(arr1)
arr1.shape = (3, 2)  # 3×2
print(arr1)
arr1.shape = (6, 1)  # 6×1
print(arr1)

# 행과 열 중에 하나만 지정하고 나머지는 -1을 지정하면, 자동으로 계산해서 치환해 준다.
arr1.shape = (-1, 1)  # 6×1
print(arr1)
arr1.shape = (2, -1)  # 2×3
print(arr1)

# 3. 1차원을 2차원으로 변환하기, reshape(행, 열) 함수
arr1 = np.array(list_value).reshape(6, 1)
print(arr1)  # 기존에 있는 행렬을 변경하기
