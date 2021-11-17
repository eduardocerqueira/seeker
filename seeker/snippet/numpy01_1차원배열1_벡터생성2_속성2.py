#date: 2021-11-17T16:51:11Z
#url: https://api.github.com/gists/3452f645eb100eb60bd9b8033d18f1b7
#owner: https://api.github.com/users/chinshin5513

import numpy as np # 별칭 주기, 앞으로는 np. 으로 사용이 가능하다.

# 1. 1차원 배열인 벡터 생성
list_value = [10, 20, 30, 40, 555]  # 파이썬 타입, 타입명: list
v1 = np.array(list_value)  # 이터러블(리스트, 튜플 등)

print("1. 벡터의 차원(dimension)", v1.ndim)  # 벡터 차원 ndim(속성)
print("2. 벡터의 차원 크기(dimension)", v1.shape)  # 값이 하나인 튜플로 반환
print("3. 벡터의 요소 갯수", v1.size)  #
print("4. 벡터의 데이터 타입", v1.dtype)  # int32 (32비트 = 4바이트 크기의 정수로 구성됨)




