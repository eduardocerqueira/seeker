#date: 2021-11-17T16:51:11Z
#url: https://api.github.com/gists/3452f645eb100eb60bd9b8033d18f1b7
#owner: https://api.github.com/users/chinshin5513

import numpy as np # 별칭 주기, 앞으로는 np. 으로 사용이 가능하다.

'''
단항 유니버셜 함수(function)
1) 각 요소에 모두 적용되는 함수
'''
# 1.더하기
arr = np.array([1, 2, 3, 4])
arr2 = arr * 2
print("1. np.add(a,a2):", np.add(arr,arr2))  # 1. np.add(a,a2): [ 3  6  9 12]
print("1. np.add(a,a2):", (arr + arr2))  # 1. np.add(a,a2): [ 3  6  9 12]

# 2. 빼기
print("2. np.subtract(a,a2):", np.subtract(arr, arr2))  # 3. np.divide(a,a2): [0.5 0.5 0.5 0.5]
print("2. np.subtract(a,a2):", (arr - arr2))  # 3. np.divide(a,a2): [0.5 0.5 0.5 0.5]

# 3. 나누기
print("3. np.divide(a,a2):", np.divide(arr, arr2))  # 2. np.subtract(a,a2): [-1 -2 -3 -4]
print("3. np.divide(a,a2):", (arr/arr2))  # 2. np.subtract(a,a2): [-1 -2 -3 -4]

# 4. 몫 연산
print("4. np.floor_divide(a,a2):", np.floor_divide(arr, arr2))  # 4. np.floor_divide(a,a2): [0 0 0 0]
print("4. np.floor_divide(a,a2):", (arr//arr2))  # 4. np.floor_divide(a,a2): [0 0 0 0]


# 5. 나머지 연산
print("5. np.mod(a,a2):", np.mod(arr, arr2))  # 4. np.floor_divide(a,a2): [0 0 0 0]
print("5. np.mod(a,a2):", (arr % arr2))  # 4. np.floor_divide(a,a2): [0 0 0 0]
print("5. np.remainder(a,a2):", np.remainder(arr, arr2))  # 4. np.floor_divide(a,a2): [0 0 0 0]

# 6. np.maximum : 최댓값 반환, np.minimum: 최솟값
print("6. 최댓값, 최솟값:", np.maximum(arr, arr2), np.minimum(arr, arr2) # 10과1 , 2와 12, 3과 3, 4와 4 비교하여 큰 값은 반환

# 7. np.greater: 인덱스별 비교해서 boolean 값으로 반화
# arr = np.array([10, 2, 3, 4])
# arr2 = np.array([1, 12, 1, 14])
# print("7, 불린 추출:", np.greater(arr, arr2))  # 10과1 , 2와 12, 3과 3, 4와 4 비교하여 큰 값은 반환하고 행렬
# print("7, 불린 추출:", np.greater(arr, arr2)) # 10과1 , 2와 12, 3과 3, 4와 4 비교하여 큰 값은 반환하고 행렬


# 요소의 값이 같은가 equal
# print("7, 불린 추출:", np.greater(arr, arr2)) # 10과1 , 2와 12, 3과 3, 4와 4 비교하여 큰 값은 반환하고 행렬
# arr = np.array([10, 2, 3, 4])
# 요소의 값이 같지 않은가 not_equal 갖지 않는다
# arr2 = np.array([1, 12, 3, 4])

# 9. np.power : 제곱(나중에 정리)
print("7. np.power(3,2)", np.power(3,2))
