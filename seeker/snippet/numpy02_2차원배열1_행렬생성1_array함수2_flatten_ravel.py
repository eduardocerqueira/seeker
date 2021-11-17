#date: 2021-11-17T16:51:11Z
#url: https://api.github.com/gists/3452f645eb100eb60bd9b8033d18f1b7
#owner: https://api.github.com/users/chinshin5513

import numpy as np # 별칭 주기, 앞으로는 np. 으로 사용이 가능하다.

'''
1. 1차원 → 2차원으로 변경할 때
    1) shape = (행, 열)
    2) reshape(행, 열) 함수 
2. 1차원 → 2차원으로 변경할 때
    1) flatten()
    2) ravel
'''
list_value = [[10, 20, 30], [1, 2, 3], [4, 5, 6]]  # 3 by 3 matrix
arr1 = np.array(list_value)
print(arr1, type(arr1))

print("1. flatten() 행 단위:", arr1.flatten())  # 2차원을 1차원으로 변환(), 행 단위 C-Style
print("2. ravel() 행 단위:", arr1.ravel())
print("3. flatten(order = 'F' ) 열 단위:", arr1.flatten(order = 'F'))  # 2차원을 1차원으로 변환(), 행 단위 F-Style
print("4. ravel() 열 단위:", arr1.ravel(order = 'F'))  # 행 단위 F-Style

# flatten과 ravel은 결과는 같지만, 내부적으로 작동하는 방식은 다르다.
list_value = [[10, 20, 30], [1, 2, 3], [4, 5, 6]]  # 3 by 3 matrix
arr1 = np.array(list_value)
print(arr1, type(arr1))

x1 = arr1.flatten()
x2 = arr1.ravel()

print("2차원 배열 수정 전.........")
print("1. flatten:", x1)  # 깊은 복사, 값 변경 안 됨
print("2. ravel:", x2)  # IN PLACE, 주소 복사, 값 변경됨

print("2차원 배열 수정 후.........")
arr1[0][0] = 100
print(arr1)
print("1. flatten:", x1)  # 깊은 복사, 값 변경 안 됨
print("2. ravel:", x2)  # IN PLACE, 주소 복사, 값 변경됨
