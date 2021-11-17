#date: 2021-11-17T16:51:11Z
#url: https://api.github.com/gists/3452f645eb100eb60bd9b8033d18f1b7
#owner: https://api.github.com/users/chinshin5513

import numpy as np # 별칭 주기, 앞으로는 np. 으로 사용이 가능하다.
'''
 2차원 배열의 색인(불린)
'''
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print("1. 벡터연산:", arr % 2 == 0)
print("1.1. 짝수값 출력:", arr[arr % 2 == 0])  # 행렬 형태가 그대로 출력되지 않고, flatten되어 출력
print("2. 짝수값이고 5보다 작거나 같은 값 출력:", arr[(arr % 2 == 0) & (arr <= 5)])  # 2. 짝수값이고 5보다 작은 값 출력: [2 4]
print("3. 짝수값이거나 5보다 작거나 같은 값 출력:", arr[(arr % 2 == 0) | (arr <= 5)])  # 3. 짝수값이거나 5보다 작은 값 출력: [1 2 3 4 5 6 8]
print("4. 홀수값이거나 5보다 큰 값 출력:", arr[~(arr % 2 == 0) | ~(arr <= 5)])  # 4. 홀수값이거나 5보다 큰 값 출력: [1 3 5 6 7 8 9]
