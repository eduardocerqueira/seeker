#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고
import seaborn as sns

'''
series
print(dir(np.char))
'add', 'array', 'array_function_dispatch', 'asarray', 'asbytes', 'bool_', 'capitalize', 'center',
'character', 'chararray', 'compare_chararrays', 'count', 'decode', 'encode', 'endswith', 'equal', 
'expandtabs', 'find', 'functools', 'greater', 'greater_equal', 'index', 'int_', 'integer', 'isalnum',
'isalpha', 'isdecimal', 'isdigit', 'islower', 'isnumeric', 'isspace', 'istitle', 'isupper', 'join',
'less', 'less_equal', 'ljust', 'lower', 'lstrip', 'mod', 'multiply', 'narray', 'ndarray', 'not_equal', 
'numpy', 'object_', 'overrides', 'partition', 'replace', 'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit',
'rstrip', 'set_module', 'split', 'splitlines', 'startswith', 'str_len', 'string_', 'strip', 'swapcase', 
'sys', 'title', 'translate', 'unicode_', 'upper', 'zfill'
'''

# 1. 첫 글자를 대문자로
arr = np.array(["hello", "WorLd", "gOOd-bYe","lOVEr"])
# 일반적인 방법(반복문)
# list = []
# for s in arr:
#     print(s.capitalize())
#     list.append(s.capitalize())
# print(np.array(list))

# 넘파이 활용하기
print("1. 첫 글자를 대문자로:", np.char.capitalize(arr))  # 1. 첫 글자를 대문자로: ['Hello' 'World' 'Good-bye' 'Lover']
print("2. 대문자:", np.char.upper(arr))  # 2. 대문자: ['HELLO' 'WORLD' 'GOOD-BYE' 'LOVER']
print("3. 소문자:", np.char.lower(arr))  # 3. 소문자: ['hello' 'world' 'good-bye' 'lover']
print("4. 대문자↔소문자:", np.char.swapcase(arr))  # 4. 대문자↔소문자: ['HELLO' 'wORlD' 'GooD-ByE' 'LoveR']
print("5. title:", np.char.title(arr))  # 5. title: ['Hello' 'World' 'Good-Bye' 'Lover']

#----------------------------------------------
arr = np.array(["hello", "world", "man"])
arr2 = np.array(['홍','길','동'])
xxx = np.char.add(arr, " ")
print("6. add:", np.char.add(arr, arr2))  # 6. add: ['hello홍' 'world길' 'man동']
print("6. add:", np.char.add(xxx, arr2)) # 6. add: ['hello 홍' 'world 길' 'man 동']  공백으로 리스트 만들기
print("7. multiply:", np.char.multiply(arr, 2))  # 7. multiply: ['hellohello' 'worldworld' 'manman']
print("8. center:", np.char.center(arr, 20, "-"))  # 8. center: ['-------hello--------' '-------world--------' '--------man---------']
print("9. rjust:", np.char.rjust(arr, 20, "-"))  # 8. center: ['-------hello--------' '-------world--------' '--------man---------']
print("10. ljust:", np.char.ljust(arr, 20, "-"))  # 8. center: ['-------hello--------' '-------world--------' '--------man---------']

#----------------------------------------------
arr = np.array(["hello", "world"])
s_encode = np.char.encode(arr, encoding="utf-8")
s_decode = np.char.decode(s_encode, encoding="utf-8")
print("11. encode:", s_encode)  # 11. encode: [b'hello' b'world'] 유니코드를 바이트로
print("12. decode:", s_decode)  # 12. decode: ['hello' 'world'] 바이트를 유니코드로

arr = np.array(["hello", "world"])
print("13.join", np.char.join(",", arr))  # 13.join ['h,e,l,l,o' 'w,o,r,l,d'] 어레이의 요소별로 적용
xxx = np.array([10, 20, 30], dtype=str)

print("13.join", np.char.join(",", xxx))  # 13.join ['1,0' '2,0' '3,0']

arr = np.array(["     hello     ", "   w o   rl   d                 "])
print("14.lstrip", np.char.lstrip(arr))  # 14.lstrip ['hello     ' 'w o   rl   d                 ']
print("14.rstrip", np.char.rstrip(arr))  # 14.rstrip ['     hello' '   w o   rl   d']
print("14.strip", np.char.strip(arr))  # 14.strip ['hello' 'w o   rl   d']

arr = np.array(["HHHhelloHHHHHH", "HHHHHHw o   rl   dAAAA"])
print("14.lstrip", np.char.lstrip(arr, "H"))  # 14.lstrip ['helloHHHHHH' 'w o   rl   dAAAA']
print("14.rstrip", np.char.rstrip(arr, "H"))  # 14.rstrip ['HHHhello' 'HHHHHHw o   rl   dAAAA']
print("14.strip", np.char.strip(arr, "H"))  # 14.strip ['hello' 'w o   rl   dAAAA']

#----------------------------------------------
arr = np.array(["aaa bbb ccc ddd", "Hello World"])
print("15.split", np.char.split(arr))  # 15.split [list(['aaa', 'bbb', 'ccc', 'ddd']) list(['Hello', 'World'])])
x, y = np.char.split(arr)
print("15.split", x, y)  # 15.split ['aaa', 'bbb', 'ccc', 'ddd'] ['Hello', 'World']
arr = np.array(["aaa/bbb/ccc/ddd", "Hello/World"])
print("15.split", np.char.split(arr, '/'))  # 15.split [list(['aaa', 'bbb', 'ccc', 'ddd']) list(['Hello', 'World'])])

arr = np.array(["John Hello is", "my name John"])
print("16. replace", np.char.replace(arr, "John", "Kim"))  # 16. replace ['Kim Hello is my name Kim']
print("16. replace", np.char.replace(arr, "John", "Kim", count=1))  # 16. replace ['Kim Hello is my name John']

print("17. find", np.char.find(arr, 'n'))  # 17. 17. find [3 3]
print("18. count", np.char.count(arr, 'e'))  # 18. count [1 1]

#----------------------------------------------
arr = np.array(["Hi", "he", "His", "She"])
print("19. startswith", np.char.startswith(arr, 'H'))  # 19. startswith [ True False  True False]
print("19. endswith", np.char.endswith(arr, 'e'))  # 19. endswith [False  True False  True]
arr2 = np.array(["Hi", "he", "His", "She"])
arr3 = np.array(["Hi", "He", "His", "she"])
print("20. equal", np.char.equal(arr, arr2))  # 20. equal [ True  True  True  True]
print("20. equal", np.char.equal(arr, arr3))  # 20. equal 20. equal [ True False  True False]
print("20. not_equal", np.char.not_equal(arr, arr2))  # 20. not_equal [False False False False]
print("20. not_equal", np.char.not_equal(arr, arr3))  # 20. not_equal [False  True False  True]

#----------------------------------------------
arr1 = np.array(["HI", "he", "His", "10"])
print("21. isalpha", np.char.isalpha(arr1))  # 21. isalpha [ True  True  True False]
print("22. isdigit", np.char.isdigit(arr1))  # 22. isdigit [False False False  True]
# ndarray는 모든 데이터를 통일시켜준다.
print("23. isupper", np.char.isupper(arr1))  # 23. isupper [ True False False False]
print("24. islower", np.char.islower(arr1))  # 24. islower [False  True False False]
# 불린 색인과 연계 필수

arr = np.array(["CAT", "Dog", "His", "She"])
print("25. str_len", np.char.str_len(arr))  # 25. str_len [3 3 3 3]
