#date: 2021-11-17T16:51:11Z
#url: https://api.github.com/gists/3452f645eb100eb60bd9b8033d18f1b7
#owner: https://api.github.com/users/chinshin5513

import numpy as np # 별칭 주기, 앞으로는 np. 으로 사용이 가능하다.
'''
 타입 변경(dtype)
 1. dtype : 벡터 안에 들어간 속성을 판단
 2. astype 함수 :
'''

# 1. int → float
arr = np.array([10, 20, 30])
print(arr, type(arr), arr.dtype)  # [10 20 30] <class 'numpy.ndarray'> int32
arr = arr.astype(float)  # astype으로 데이터 타입을 바꾸기
print(arr, type(arr), arr.dtype)  # [10. 20. 30.] <class 'numpy.ndarray'> float64

# 2. float → int
arr = np.array([10, 20, 30], dtype = float)
print(arr, type(arr), arr.dtype)  # [10. 20. 30.] <class 'numpy.ndarray'> float64
arr = arr.astype(int)  # astype으로 데이터 타입을 바꾸기
print(arr, type(arr), arr.dtype)  # [10 20 30] <class 'numpy.ndarray'> int32

# 3. int → bytes, str
arr = np.array([10, 20, 30])
print(arr, type(arr), arr.dtype)  # [10 20 30] <class 'numpy.ndarray'> int32
# arr = arr.astype(str)  # astype으로 데이터 타입을 바꾸기(문자열)
# print(arr, type(arr), arr.dtype)  # ['10' '20' '30'] <class 'numpy.ndarray'> <U11
arr = arr.astype(bytes)  # astype으로 데이터 타입을 바꾸기(바이트)
print(arr, type(arr), arr.dtype)  # [b'10' b'20' b'30'] <class 'numpy.ndarray'> |S11

# 4. str → int
arr = np.array([10, 20, 30], dtype=str)
print(arr, type(arr), arr.dtype)  # [['10' '20' '30'] <class 'numpy.ndarray'> <U2
arr = arr.astype(int)  # astype으로 데이터 타입을 바꾸기(바이트)
print(arr, type(arr), arr.dtype)  # [10 20 30] <class 'numpy.ndarray'> int32