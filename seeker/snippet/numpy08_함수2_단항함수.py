#date: 2021-11-17T16:51:11Z
#url: https://api.github.com/gists/3452f645eb100eb60bd9b8033d18f1b7
#owner: https://api.github.com/users/chinshin5513

import numpy as np # 별칭 주기, 앞으로는 np. 으로 사용이 가능하다.

'''
단항 유니버셜 함수(function)
1) 각 요소에 모두 적용되는 함수
'''
# 1. np.abs
arr = np.array([3.15554, -2.355211, 0, -5.33217, 8.3344442])
print("1. 절댓값:", np.abs(arr))
print("1. 절댓값:", np.absolute(arr))

# 2. np.around
print("2. 0.5 기준으로 반올림:", np.around(arr))

# 3. np.round
print("3. 반올림:", np.around(arr, 0))  # 반올림해서 정수로 만들기
print("3. 반올림:", np.around(arr, 2))  # 자릿 수 지정하여 반올림

# 4. np.rint : 가장 가까운 정수로 올림 또는 내림
print("4. 정수 반올림:", np.rint(arr))  # 4. 정수 반올림: [ 3. -2.  0. -5.  8.]

# 5. np.ceil : 같거나 큰, 가장 작은 정수값(SQL), 올림
print("5. ceil:", np.ceil(arr))  # 5. ceil: [ 4. -2.  0. -5.  9.]

# 6. np.floor : 같거나 큰, 가장 작은 정수값 내림
print("6. floor:", np.floor(arr))  # 6. floor: [ 3. -3.  0. -6.  8.]

# 7. np.trunc : 같거나 큰, 가장 작은 정수값 내림
print("6. trunc:", np.trunc(arr))  # 6. floor: [ 3. -3.  0. -6.  8.]

# 8. np.sqrt : 제곱근
arr = np.array([1, 4, 9, 16, 25])
print(np.sqrt(arr))
