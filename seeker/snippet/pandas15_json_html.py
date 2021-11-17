#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고
import seaborn as sns

'''
json
 1. 읽기
    pd.read_json("파일경로", 옵션)
 2. 쓰기
    pd.to_json("파일경로")   
'''

df = pd.read_json(".\data\my.json")
print("1. 기본: \n", df.head())

'''
html
 1. 읽기
    pd.read_html(url경로) → 테이블의 태그를 읽는다.
 2. 쓰기
    pd.to_json("파일경로")   
'''
table = pd.read_html("https://www.seoul.go.kr/coronaV/coronaStatus.do")
print(len(table))
print(table[0].head())
print(table[1].head())
print(table[2].head())
