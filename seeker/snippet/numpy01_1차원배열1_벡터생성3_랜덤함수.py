#date: 2021-11-17T16:51:11Z
#url: https://api.github.com/gists/3452f645eb100eb60bd9b8033d18f1b7
#owner: https://api.github.com/users/chinshin5513

import numpy as np # 별칭 주기, 앞으로는 np. 으로 사용이 가능하다.

# 랜덤함수
r1 = np.random.random()  # []는 inclusive, ()는 exclusive
print("1. 랜덤함수:", r1)  # 0<= r1 < 1 표기법은 [0.0, 1.0)
r2 = np.random.random(5)  # size: 무작위로 값을 5개 받아옴
print("2. 다섯 개 추출", r2)
r3 = np.random.rand()  # [0, 1) 사이의 균등분포에서 랜덤한 값을 추출함(통계학: 균등분포)
print("3. 균등분포 추출", r3)
r4 = np.random.randn(5)  # 평균이 0이고 표준편차가 1인 정규분포에서 랜덤한 값을 추출함(통계학: 정규분포)
print("4. 표준정규분포 추출", r4)
r5 = np.random.randint(1, 11, 3, np.int64)  # [1, 11) 범위를 지정하고 몇 개의 정수를 추출하는 방법(중복 추출 가능, 독립시행), 값 형식 지정
print("5. 범위 지정하고 추출", r5)
r5 = np.random.randint(low=1, high=11, size=3, dtype=np.int64)  # [1, 11) 범위를 지정하고 몇 개의 정수를 추출하는 방법(중복 추출 가능, 독립시행), 값 형식 지정
print("5. 범위 지정하고 추출", r5)  # low 디폴트는 0, 숫자가 하나면 0부터 시작

# 0부터 6까지 랜덤하게 3개 추출
r6 = np.random.randint(low=0, high=6, size=3)
print(r6)

#
r7 = np.random.choice(['A', 'B', 'C'])  # 랜덤하게 선택하기
print("6. 랜덤하게 선택", r7)

list_value = [6, 2, 6, 8, 9,10] # 순서를 섞어주기
np.random.shuffle(list_value)
print(list_value)  # IN PLACE