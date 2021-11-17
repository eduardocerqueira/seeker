#date: 2021-11-17T16:51:11Z
#url: https://api.github.com/gists/3452f645eb100eb60bd9b8033d18f1b7
#owner: https://api.github.com/users/chinshin5513

import numpy as np # 별칭 주기, 앞으로는 np. 으로 사용이 가능하다.

'''
단항 유니버셜 함수(function)
1) 각 요소에 모두 적용되는 함수
'''
# 1.곱 계산
arr = np.array([1, 2, 3, 4])
print("1. np.prod(arr):", np.prod(arr))
arr = np.array([[1, 2], [3, 4], [5, 6]])
print("1. np.prod(arr):", np.prod(arr))  # axis 없으면 스칼라값
print("1. np.prod(arr, keepdims=True):", np.prod(arr, keepdims=True))  # 2차원 행렬 유지
print("1. np.prod(arr, axis=0):", np.prod(arr, axis=0))  # [15 48]
print("1. np.prod(arr, axis=1):", np.prod(arr, axis=1))  # [ 2 12 30]

# 2. np.nanprod: nan을 1로 생각하고 처리
arr = np.array([[1, 2], [3, 4], [5, np.nan]])  # np.nan은 Null을 의미
print("2. np.prod(nan):", np.prod(arr))  # Null 값을 연산해서 Null이 나옴
print("2. np.nanprod(nan):", np.nanprod(arr))  # 1. np.nanprod(nan): 120.0

# 3. np.cumprod: 행렬별 누적 곱
arr = np.array([[1, 2], [3, 4], [5, 6]])
print("3. np.cumprod(arr, axis=0):", np.cumprod(arr, axis=0))  # [[ 1  2] [ 3  8]  [15 48]]
print("3. np.cumprod(arr, axis=1):", np.cumprod(arr, axis=1))  # [[ 1  2] [ 3 12]  [ 5 30]]

# 4. np.sum: 누적 합
arr = np.array([[1, 2], [3, 4], [5, 6]])
print("4. np.sum(arr, axis=0):", np.sum(arr, axis=0))  # [ 9 12]
print("4. np.sum(arr, axis=1):", np.sum(arr, axis=1))  # [ 3  7 11]

# 5. np.nansum: nan을 0으로 처리
arr = np.array([[1, 2], [3, 4], [5, np.nan]])  # np.nan은 Null을 의미
print("5. np.sum(arr, axis=0):", np.sum(arr, axis=0))  # [ 9. nan]
print("5. np.nansum(arr, axis=1):", np.nansum(arr, axis=1))  # [3. 7. 5.]

# 3. np.cumsum: 행렬별 누적 곱
arr = np.array([[1, 2], [3, 4], [5, 6]])
print("6. np.cumsum(arr, axis=0):", np.cumsum(arr, axis=0))  # [[ 1  2] [ 4  6] [ 9 12]]
print("6. np.cumsum(arr, axis=1):", np.cumsum(arr, axis=1))  # [[ 1  3] [ 3  7] [ 5 11]]

# -----------통계관련 함수---------------------------------------------------
# 1. np.mean: 평균
arr = np.array([1, 2, 3, 4])
print("1, np.mean:", np.mean(arr))  # 1, np.mean: 2.5
arr = np.array([[1, 2], [3, 4], [5, 6]])
print("1, np.mean:", np.mean(arr, axis=0))  # [3. 4.]
print("1, np.mean:", np.mean(arr, axis=1))  # [1.5 3.5 5.5]

# 2. np.var: 분산
arr = np.array([1, 2, 3, 4])
print("2. np.var:", np.var(arr))  # 1.25
arr = np.array([[1, 2], [3, 4], [5, 6]])
print("2. np.var:", np.var(arr, axis=0))  # [2.66666667 2.66666667]
print("2. np.var:", np.var(arr, axis=1))  # [0.25 0.25 0.25]

# 3. np.std: 표준편차
arr = np.array([1, 2, 3, 4])
print("3. np.std:", np.std(arr))  # 1.118033988749895
arr = np.array([[1, 2], [3, 4], [5, 6]])
print("3. np.std:", np.std(arr, axis=0))  # [1.63299316 1.63299316]
print("3. np.std:", np.std(arr, axis=1))  # [0.5 0.5 0.5]

# 4. np.max, np.min: 최댓값, 최솟값
arr = np.array([1, 2, 3, 4])
print("4. np.max, np.min:", np.max(arr), np.min(arr))  # 4 1
arr = np.array([[1, 2], [3, 4], [5, 6]])
print("4. np.max, np.min:", np.max(arr, axis=0), np.min(arr, axis=0))  # [5 6] [1 2]
print("4. np.max, np.min:", np.max(arr, axis=1), np.min(arr, axis=1))  # [2 4 6] [1 3 5]

# 5. np.argmax, np.argmin: 최댓값 위치, 최솟값 위치
arr = np.array([1, 2, 3, 4])
print("5. np.argmax, np.argmin:", np.argmax(arr), np.argmin(arr))  #  3 0
arr = np.array([[1, 2], [3, 4], [5, 6]])
print("5. np.argmax, np.argmin:", np.argmax(arr, axis=0), np.argmin(arr, axis=0))  # [2 2] [0 0]
print("5. np.argmax, np.argmin:", np.argmax(arr, axis=1), np.argmin(arr, axis=1))  # [1 1 1] [0 0 0]
# 값이 저장되어 있는 리스트의 위치를 반환함

# 6. np.sort: 정렬
arr = np.array([4, 2, 1, 5])
print("6. np.sort", np.sort(arr))  #  3 0
arr = np.array([[1, 5, 2], [40, 12, 54], [40, 50, 6]])
print("6. np.sort", np.sort(arr, axis=0))  # [[ 1  5  2] [40 12  6] [40 50 54]]
print("6. np.sort", np.sort(arr, axis=1))  # [[ 1  2  5] [12 40 54] [ 6 40 50]]
print("6. np.sort", np.sort(arr, axis=None))  # [ 1  2  5  6 12 40 40 50 54]