#date: 2021-11-17T16:51:11Z
#url: https://api.github.com/gists/3452f645eb100eb60bd9b8033d18f1b7
#owner: https://api.github.com/users/chinshin5513

import numpy as np # 별칭 주기, 앞으로는 np. 으로 사용이 가능하다.

# ones: 모든 요소를 1로 채우는 벡터
r1 = np.ones(5)  # size 필수
print("1. ones(5):", r1, r1.dtype)  # 저장된 속성은 dtype => 디폴트 타입은 실수(float)

r1 = np.ones(5, dtype=int)  # 저장 타입을 바꾸는 방법, np.int DeprecationWarning : 과거에는 사용했지만, 현재는 사용을 권장하지 않음
print("2. ones(5):", r1, r1.dtype)  # 저장된 속성은 dtype => 디폴트 타입은 실수(float)

# zeros: 모든 요소를 0으로 채우는 벡터
r1 = np.zeros(5)
print("3. zeros(5):", r1, r1.dtype)  # 저장된 속성은 dtype => 디폴트 타입은 실수(float)
r1 = np.zeros(5, dtype=int)
print("3. zeros(5):", r1, r1.dtype)  # 저장된 속성은 dtype => 디폴트 타입은 실수(float)

# empty : 임의의 값으로 채우기, 초기화의 의미가 강함
r1 = np.empty(5, dtype=int)
print("3. empty(5):", r1, r1.dtype)  # 아무 숫자나 나온다.

# full : 내가 지정한 값으로 채우기
r1 = np.full(5, 3, dtype=int)
print("3. full(5, 10):", r1, r1.dtype)  # 3. full(5, 10): [3 3 3 3 3] int32
