#date: 2021-11-17T16:51:11Z
#url: https://api.github.com/gists/3452f645eb100eb60bd9b8033d18f1b7
#owner: https://api.github.com/users/chinshin5513

import numpy as np # 별칭 주기, 앞으로는 np. 으로 사용이 가능하다.

'''
기본 파이썬의 얕은 복사, 깊은 복사
깊은 복사
[:], list(), arr.copy()
얕은 복사
변수 할당

넘파잉에서 얕은 복사, 깊은 복사
얕은 복사 arr[:]
깊은 복사 arr.copy, np.copy(arr)
'''

# 파이썬에서는
arr = list(range(10))
print("1. 원본:", arr)  # 1. 원본: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
arr2 = arr
arr2[0] = 100
print("1.1. 얕은 복사:", arr)  # 1. 얕은 복사: [100, 1, 2, 3, 4, 5, 6, 7, 8, 9]

arr = list(range(10))
# copy_arr = arr.copy()
# copy_arr = arr[:])
copy_arr = list(arr)
copy_arr[0] = 100
print("1.2. 깊은 복사:", arr)  # 1.2. 깊은 복사: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print("---------넘파이 시작------------")  # 1.2. 깊은 복사: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


arr = np.arange(10)
print("10. ndarray 원본:", arr)  # 10. ndarray 원본: [0 1 2 3 4 5 6 7 8 9]
copy_arr = arr[:]  # 얕은 복사
copy_arr[0] = 100
print(arr)

arr = np.arange(10)
print("10. ndarray 원본:", arr)  # 10. ndarray 원본: [0 1 2 3 4 5 6 7 8 9]
copy_arr = np.copy(arr)  # 깊은 복사
# copy_arr = ar.copy()  # 깊은 복사
copy_arr[0] = 100
print("11. 깊은 복사:", arr)  # [0 1 2 3 4 5 6 7 8 9]