#date: 2021-11-17T16:51:11Z
#url: https://api.github.com/gists/3452f645eb100eb60bd9b8033d18f1b7
#owner: https://api.github.com/users/chinshin5513

import numpy as np # 별칭 주기, 앞으로는 np. 으로 사용이 가능하다.

# arrange : 파이썬의 range 함수
# 차이점: 파이썬은 리스트, arrange는 쉼표가 없는 ndarray로 반환
# np.arange( start, stop(필수), step, dtype (디폴트는 정수) )
v1 = np.arange(10)
print("1. np.arange(10)", v1)  # 1. np.arange(10) [0 1 2 3 4 5 6 7 8 9]

v1 = np.arange(1, 10, 2)
print("2. np.arange(1, 10, 2)", v1)  # 2. np.arange(1, 10, 2) [1 3 5 7 9]

v1 = np.arange(10.0)
print("3. np.arange(1, 10, 2)", v1)  # 3. np.arange(1, 10, 2) [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]


