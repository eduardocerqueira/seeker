#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고
import seaborn as sns

'''
series.str.문자열함수(arr) : 활용도가 매우 높다.
'''

df = sns.load_dataset("mpg")
print(df.head())

'''
print(dir(df["name"].str))
'capitalize', 'casefold', 'cat', 'center', 'contains', 'count', 'decode', 'encode', 'endswith', 'extract',
'extractall', 'find', 'findall', 'fullmatch', 'get', 'get_dummies', 'index', 'isalnum', 'isalpha', 
'isdecimal', 'isdigit', 'islower', 'isnumeric', 'isspace', 'istitle', 'isupper', 'join', 'len', 'ljust', 
'lower', 'lstrip', 'match', 'normalize', 'pad', 'partition', 'repeat', 'replace', 'rfind', 'rindex', 
'rjust', 'rpartition', 'rsplit', 'rstrip', 'slice', 'slice_replace', 'split', 'startswith', 'strip', 
'swapcase', 'title', 'translate', 'upper', 'wrap', 'zfill'
'''
data = df["name"].head(10)
print("1. 첫 글자를 대문자로: \n", data.str.capitalize())
print("2. 모두 대문자로: \n", data.str.upper())
print("3. 모두 소문자로: \n", data.str.lower())
print("4. 대문자↔소문자: \n", data.str.swapcase())
print("5. 단어 첫 글자를 대문자로: \n", data.str.title())
print("6. encode: \n", data.str.encode("utf-8"))
b_data = data.str.encode("utf-8")
print("7. decode: \n", b_data.str.decode("utf_8"))

print("8. join: \n", data.str.join("~~~~"))  # 문자 사이마다 연결해줌

series_x = pd.Series(["     Hello, World         ","xyz      ","        xyz"])
print("9. strip: \n", series_x.str.strip())  #
print("10. lstrip: \n", series_x.str.lstrip())  #
print("11. rstrip: \n", series_x.str.rstrip())  #

series_x = pd.Series(["xxxxxxxxHello, Worldxxxxxxxxxxxxx","xyzxxxxxxxxxxxxx","xxxxxxxxxxxxyz"])
print("12. strip: \n", series_x.str.strip('x'))  # Hello, World, yz, yz
print("13. lstrip: \n", series_x.str.lstrip('x'))  # Hello, Worldxxxxxxxxxxxxx, yzxxxxxxxxxxxxx, yz
print("14. rstrip: \n", series_x.str.rstrip('x'))  # xxxxxxxxHello, World, xyz, xxxxxxxxxxxxyz

series_x = pd.Series(["hello", "world", "Be", "happy"])
print("15. startswith: \n", series_x.str.startswith('h'))  # Boolean
print("16. endswith: \n", series_x.str.startswith('y'))  # Boolean
print("17. contains: \n", series_x.str.contains('o'))  # Boolean
print("18. 특정문자 포함된 값으로 반환: \n", series_x[series_x.str.contains('o')])  # str

print("19. find: \n", series_x.str.find('o'))  # 있으면 인덱스 숫자, 없으면 -1

print("20. ljust: \n", series_x.str.ljust(10, '_'))  # 왼쪽에 위치
print("21. rjust: \n", series_x.str.rjust(10, '_'))  # 오른쪽에 위치
print("22. center: \n", series_x.str.center(10, '_'))  # 중앙에 위치

print("23. islower: \n", series_x.str.islower())  # 전부 소문자인지 묻기
print("24. isupper: \n", series_x.str.isupper())  # 전부 대문자인지 묻기
print("25. isalpha: \n", series_x.str.isalpha())  # 전부 문자인지 묻기
print("26. isdigit: \n", series_x.str.isdigit())  # 전부 숫자인지 묻기
