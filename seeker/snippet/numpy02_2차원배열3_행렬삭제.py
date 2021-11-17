#date: 2021-11-17T16:51:11Z
#url: https://api.github.com/gists/3452f645eb100eb60bd9b8033d18f1b7
#owner: https://api.github.com/users/chinshin5513

import numpy as np # 별칭 주기, 앞으로는 np. 으로 사용이 가능하다.
'''
 2차원 배열의 삭제
  1. np.delete 함수를 사용
  2. axis 지정을 하지 않으면, 1차원으로 반환함 (flatten)
  
'''
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr)

# 1. np.delete(arr, idx, axis = 0/1)
# copy_arr = np.delete(arr, 2)  # axis 지정하지 않으면 flatten 되어 반환됨
# print(arr)
# copy_arr = np.delete(arr, 1, axis=0)  # 행 삭제
# print(copy_arr)
copy_arr = np.delete(arr, 1, axis=1)  # 열 삭제
print(copy_arr)

