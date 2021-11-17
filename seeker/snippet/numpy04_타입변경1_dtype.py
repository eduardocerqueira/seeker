#date: 2021-11-17T16:51:11Z
#url: https://api.github.com/gists/3452f645eb100eb60bd9b8033d18f1b7
#owner: https://api.github.com/users/chinshin5513

import numpy as np # 별칭 주기, 앞으로는 np. 으로 사용이 가능하다.
'''
 타입 변경(dtype)
 1. dtype : 벡터 안에 들어간 속성을 판단
 2. astype 함수 :
'''

# # 1. int → float
# arr = np.array([10, 20, 30], dtype=int)
# print(arr, type(arr), arr.dtype)  # [10 20 30] <class 'numpy.ndarray'> int32
# f_arr = np.array([10, 20, 30], dtype=float)
# print(f_arr, type(f_arr), f_arr.dtype)  # [10. 20. 30.] <class 'numpy.ndarray'> float64
# f_arr = np.array([10, 20, 30], dtype='f')
# print(f_arr, type(f_arr), f_arr.dtype)  # [10. 20. 30.] <class 'numpy.ndarray'> float32
# f_arr = np.array([10, 20, 30], dtype='f8')
# print(f_arr, type(f_arr), f_arr.dtype)  # [10. 20. 30.] <class 'numpy.ndarray'> float64

# 2. float → int
arr = np.array([10, 20, 30], dtype=float)
print(arr, type(arr), arr.dtype)  # [10 20 30] <class 'numpy.ndarray'> int32
i_arr = np.array([10, 20, 30], dtype=int)

# 3. int → str
arr = np.array([10, 20, 30])
s_arr = np.array([10, 20, 30], dtype=str)  # (유니코드 변환)
print(s_arr, s_arr.dtype)  # ['10' '20' '30'] <U2 유니코드 문자열로 변환됨
s_arr = np.array([10, 20, 30], dtype=bytes)  # dtype = np.string (바이트 변환)
print(s_arr, s_arr.dtype)  # [b'10' b'20' b'30'] |S2 바이트로 변화됨, 인터넷 네트워크 크롤링

# 문자열 종류에서 유니코드 ↔ 바이트 간 상호 변환 공부할 것
arr = np.array(['10', '20'])
print(arr, arr.dtype)
i_arr = np.array(['10', '20'], dtype=int)
print(i_arr, i_arr.dtype)