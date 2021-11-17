#date: 2021-11-17T16:51:11Z
#url: https://api.github.com/gists/3452f645eb100eb60bd9b8033d18f1b7
#owner: https://api.github.com/users/chinshin5513

import numpy as np # 별칭 주기, 앞으로는 np. 으로 사용이 가능하다.

'''
 2차원 배열의 추가방법
  1. np.append 함수를 사용
  2. axis 지정을 하지 않으면, 1차원으로 반환함 (flatten)
     따라서 기존배열의 열 갯수와 일치하지 않아도 된다.
  3. axis 지정하면 반드시 기존 배열의 열 갯수와 일치해야 한다
'''
# 1. np.append(arr, 값, axis = 0/1)
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr)

# arr_copy = np.append(arr, [[7, 8, 9]], axis=0)  # 행 단위로 붙이기
# print(arr_copy, type(arr_copy))


arr_copy = np.append(arr, [[100], [200]], axis=1)  # 열 단위로 붙이기
print(arr_copy, type(arr_copy))  # 차원에 대한 이해가 필요함

# 2. np.insert(arr, idx, axis = 0)
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr)
# arr_copy = np.insert(arr, 0, 100, axis = 1) # arr 배열, 0번째 열에 100을 집어넣는다. (100을 하나만 써도 행마다 들어감)
# print(arr_copy)
# arr_copy = np.insert(arr, 0, 100, axis = 0) # arr 배열, 0번째 행에 100을 집어넣는다. (100을 하나만 써도 열마다 들어감)
# print(arr_copy)
# arr_copy = np.insert(arr, 0, 300) # axis를 미지정하면, flatten 되어서 출력됨
# print(arr_copy)


'''
이러한 작업을 브로드캐스팅 작업이라고 함
[0 1 2 3 4] * 4 → [0 1 2 3 4] * [4 4 4 4 4] → [ 0 4 8 12 16] 브로드캐스팅
벡터의 행렬을 맞추기 위해 자동적으로 길이를 늘려주는 작업
'''


