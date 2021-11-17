#date: 2021-11-17T16:51:11Z
#url: https://api.github.com/gists/3452f645eb100eb60bd9b8033d18f1b7
#owner: https://api.github.com/users/chinshin5513

import numpy as np # 별칭 주기, 앞으로는 np. 으로 사용이 가능하다.

# 1. np.nonzero : 0이 아닌 값의 인덱스(idx)를 반환
arr = np.nonzero([0, 0, 3, 4, 5])
print("1. nonzero:", arr)  # (array([2, 3, 4], dtype=int64),)

# 2. np.where : 3항 연산자: 참 if 조건식 else 거짓
arr = np.arange(10)
print(arr)  # [0 1 2 3 4 5 6 7 8 9]
print("2. np.where: ", np.where(arr < 5, arr, arr * 10))  # 1. np.where:  [ 0  1  2  3  4 50 60 70 80 90]

# 3. np.where : 값의 위치값을 반환함
arr = np.array([9, 3, 6, 87])
print("3. np.where:", np.where(arr == 3))  # 3. np.where: (array([1], dtype=int64),)
arr = np.array([[9, 3, 6, 87],[3, 4, 5, 6]])
print("3. np.where:", np.where(arr == 3))  # 3. np.where: (array([0, 1], dtype=int64), array([1, 0], dtype=int64))

# 4. np.unique : 중복된 값을 제거 후, 벡터로 변환
arr = np.array([9, 3, 6, 87, 3, 45, 9, 1, 10])
print("4. np.unique:", np.unique(arr))  # 4. np.unique: [ 1  3  6  9 10 45 87]
arr = np.array([[9, 3, 6, 87], [3, 4, 5, 6]])
print("4. np.unique:", np.unique(arr))  # 4. flatten 되어 벡터로 산출, 4. np.unique: [ 3  4  5  6  9 87]

# 5. np.choose : 팬시 색인하고 비슷함
arr = np.array(["A", "B", "C", "D", "E"])
print("5. A와 B, E 색인", arr[[0, 1, 4]])  # 5. A와 B, E 색인 ['A' 'B' 'E']
print("5. A와 B, E 색인", np.choose([0, 1, 4], arr))  # 5. A와 B, E 색인 ['A' 'B' 'E']

# 6. np.choose : 다중 조건 지정
x = np.arange(10)
condlist = [x<3, x>5]
choicelist = [x, x**2]
print("6. select:", np.select(condlist, choicelist))  # 6. select: [ 0  1  2  0  0  0 36 49 64 81]
print("6. select:", np.select(condlist, choicelist, default = 1))  # 6. select: [ 0  1  2  1  1  1 36 49 64 81]

# 7. np.all, np. any → boolean 값
print("7. np.all:", np.all([True, True]))  # 7. np.all: True  모든 값이 참이냐
print("7. np.all:", np.all([True, False]))  # 7. np.all: False  모든 값이 참이냐
print("7. np.any:", np.any([True, True]))  # 7. np.any: True
print("7. np.any:", np.any([True, False]))  # 7. np.any: True
print("7. np.any:", np.any([False, False]))  # 7. np.any: False

# 응용: 모든 문자열이 대문자인가?
arr = np.array(["Hi", "HE", "His", "She"])
print(np.char.isupper(arr))  # [False  True False False]
print(np.all(np.char.isupper(arr)))  # False
print(np.any(np.char.isupper(arr)))  # True

# 응용: 문자열의 길이가 세 글자 이상인가?
print(np.all(np.where(np.char.str_len(arr) >= 3, True, False)))
print(np.all([len(arr) >= 3 for x in arr]))
arr = np.array(["CAT", "Dog", "His", "She"])
print(np.all(np.where(np.char.str_len(arr) >= 3, True, False)))
print(np.all(np.char.str_len(arr) >= 3))
print(np.all([len(arr) >= 3 for x in arr]))

# 8. np.fromstring → split 기능을 포함하는 함수
result = np.fromstring('10 20', sep=' ')
print(result, result.dtype)  # [10. 20.] float64
result = np.fromstring('10 20', dtype=int, sep=' ')
print(result, result.dtype)  # [10 20] int32

# 응용 합을 계산 '10 20 30'
result = np.fromstring('10 20 30', dtype=int, sep=' ')
print(np.sum(result)) # 60

# 9. 전치 : 행과 열을 바꾸어서 표현하는 것(transpose)
arr = np.arange(15).reshape(3, 5)
print(arr)
'''
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]]
'''
print("전치(속성):", arr.T)
print("전치(함수):", np.transpose(arr))
'''
[[ 0  5 10]
 [ 1  6 11]
 [ 2  7 12]
 [ 3  8 13]
 [ 4  9 14]]
 '''

# 10. 내적 : 행렬에서의 곱셈 (np.dot)
arr1 = np.arange(1, 5).reshape(2, 2)
print("내적:", np.dot(arr1, arr1))

