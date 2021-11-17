#date: 2021-11-17T16:51:11Z
#url: https://api.github.com/gists/3452f645eb100eb60bd9b8033d18f1b7
#owner: https://api.github.com/users/chinshin5513

import numpy as np # 별칭 주기, 앞으로는 np. 으로 사용이 가능하다.

# 1. numpy 버전 확인
# print("1. numpy 버전 확인", np.__version__)

# 2. 1차원 배열인 벡터 생성
list_value = [10, 20, 30, 40]  # 파이썬 데이터
vector01 = np.array(list_value)  # 이터러블(리스트, 튜플 등)
print(vector01, type(vector01))  # [10 20 30 40] <class 'numpy.ndarray'> , 콤마가 없음

list_value2 = [x for x in range(0, 10)]  # 리스트 컴프리헨션으로 리스트 제작
vector02 = np.array(list_value2)
print(vector02)

list_value3 = [10, 20, 30, '40']  # 하나만 문자여도
vector03 = np.array(list_value3)
print(vector03)  # 넘파이로 넘어오면서 문자로 변경됨

# 3. 자동 형변환
int_float_value = [1, 2, 3, 4, 5.5]  # 실수가 있을 경우
vector04 = np.array(int_float_value)
print(vector04)  # 모두 실수로 형변환이 됐음(float)
# 넘파이의 자동 형변환은 범위가 조금 더 큰 자료형으로 변환된다. 정수 → 실수 → 문자열
