#date: 2021-11-17T16:51:11Z
#url: https://api.github.com/gists/3452f645eb100eb60bd9b8033d18f1b7
#owner: https://api.github.com/users/chinshin5513

import numpy as np # 별칭 주기, 앞으로는 np. 으로 사용이 가능하다.

'''
색인: 기존 파이썬 문법과 동일
팬시 색인: 넘파이에만 있음, 정수배열색인 → [idx, idx2 ... ]
불린(boolean) 색인: 논리값으로 해당되는 값을 찾는 것 → [ True, False, True ], True만 찾음
          원본의 크기와 동일하게 주어야 한다. 불린 색인이 가능한 이유는 벡터이기 때문이다.
'''

v1 = np.arange(0, 5)  # [0 1 2 3 4]
v2 = np.arange(0, 5) * 3

print("1. 원본:", v1)  # 1. 원본: [0 1 2 3 4]
print("2. 벡터연산: 벡터와 스칼라(+)", (v1+2))  # 2. 벡터연산: 벡터와 스칼라(+) [2 3 4 5 6]
print("2. 벡터연산: 벡터와 벡터(+)", (v1+v2))  # 2. 벡터연산: 벡터와 벡터(+) [ 0  4  8 12 16] 사칙연산 모두 가능
print("3. 벡터연산: 벡터와 벡터(비교연산)", v1 == 0)  # 3. 벡터연산: 벡터와 벡터(비교연산) [ True False False False False]
print("3. 벡터연산: 벡터와 벡터(비교연산)", v1 >= 3)  # 3. 벡터연산: 벡터와 벡터(비교연산) [False False False  True  True]

print("4. 불린색인:", v1[[True, False, True, False, False]])  # 4. 불린색인: [0 2], 트루 값만 뽑아서 ndarray로 만들어줌
# 우리는 논리값을 직접 쓰지 않고, 논리값에 쓰는 연산자로 쓴다.

v1 = np.arange(10, 20)
print("짝수만 묻기:", v1[v1 % 2 == 0])  # 짝수만 묻기: [10 12 14 16 18]
print("15보다 큰 수 묻기:", v1[v1 >= 15])  # 15보다 큰 수 묻기: [15 16 17 18 19]

'''
넘파이의 논리 연산자
AND &
OR |
NOT ~
'''
# 논리연산자가 반환되는 순서를 기억할 것
# v1에서 짝수값이고 15보다 큰 값은?
print(v1[(v1 % 2 == 0) & (v1 >= 15)])  # [16 18]
# v1에서 짝수값이거나 15보다 큰 값은?
print(v1[(v1 % 2 == 0) | (v1 >= 15)])  # [10 12 14 15 16 17 18 19]
# v1에서 홀수값이거나 15보다 큰 값은?
print(v1[~(v1 % 2 == 0) | (v1 >= 15)])  # [11 13 15 16 17 18 19]
print(v1[(v1 % 2 != 0) | (v1 >= 15)])  # [11 13 15 16 17 18 19]
