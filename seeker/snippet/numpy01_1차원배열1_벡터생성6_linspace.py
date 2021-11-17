#date: 2021-11-17T16:51:11Z
#url: https://api.github.com/gists/3452f645eb100eb60bd9b8033d18f1b7
#owner: https://api.github.com/users/chinshin5513

import numpy as np # 별칭 주기, 앞으로는 np. 으로 사용이 가능하다.

# linspace: 균등하게 값을 반환하는 작업
# np.linspace(시작, 값, 나누어서 반환하고 싶은 요소의 수, endpoint = True/False)
v1 = np.linspace(0, 1, 3)  # 처음, 끝 모두 포함됨, [0.  0.5 1. ] 실수가 디폴트
print(v1)

v1 = np.linspace(0, 0.9, 10)  # 처음, 끝 모두 포함됨, [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9] 실수가 디폴트
print(v1)

v1 = np.linspace(0, 0.9, 10, endpoint=True)  # 처음, 끝 모두 포함됨
print(v1)
v1 = np.linspace(0, 0.9, 10, endpoint=False)  # 처음 포함, 끝 포함 안 됨
print(v1)
