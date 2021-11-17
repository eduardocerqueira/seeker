#date: 2021-11-17T16:51:11Z
#url: https://api.github.com/gists/3452f645eb100eb60bd9b8033d18f1b7
#owner: https://api.github.com/users/chinshin5513

import numpy as np # 별칭 주기, 앞으로는 np. 으로 사용이 가능하다.

'''
1. 1차원 → 2차원으로 변경할 때
    1) shape = (행, 열)
    2) reshape(행, 열) 함수 
2. 2차원 → 1차원으로 변경할 때
    1) flatten()
    2) ravel
'''
# 1. 2차원 배열의 행렬 생성
list_value = [[10, 20, 30], [1, 2, 3], [4, 5, 6]]  # 3 by 3 matrix
arr1 = np.array(list_value)
print(arr1, type(arr1))

print("1. 벡터의 차원(dimension)", arr1.ndim)  # 2차원 ndim(속성) 2
print("2. 벡터의 차원 크기(dimension)", arr1.shape)  # 튜플로 변환 (3, 3)
print("3. 벡터의 요소 갯수", arr1.size)  # 9
print("4. 벡터의 데이터 타입", arr1.dtype)  # int32 (32비트 = 4바이트 크기의 정수로 구성됨)

'''
int 8 : 1 byte : -128 ~ 127 
int16 : 2 byte : -32768 ~ 32767 
int32 : 4 byte : -2147483648 ~ 214748367 
int62 : 8 byte :
'''

