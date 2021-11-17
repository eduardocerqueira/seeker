#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고
import seaborn as sns

'''
csv 파일
 1. 읽기
    pd.read_csv("파일경로", 옵션)
 2. 쓰기
    pd.to_csv("파일경로")   
'''

df = pd.read_csv(".\data\msft.csv")
print("1. 기본: \n", df.head())

df = pd.read_csv(".\data\msft.csv", index_col =0)
print("2. 특정 컬럼을 인덱스로 변경: \n", df.head())

df = pd.read_csv(".\data\msft.csv", usecols=["Date", "High"])
print("3. 특정 컬럼만 반환: \n", df.head())

df = pd.read_csv(".\data\msft.csv", nrows=2)
print("4. 행 갯수 지정: \n", df.head())

df = pd.read_csv(".\data\msft.csv", skipfooter=2)
print("5. 끝행 갯수 n개 제외하기: \n", df.tail())

# 널 처리
df = pd.read_csv(".\data\company3.csv", na_filter=True)
df = pd.read_csv(".\data\company3.csv", na_filter=False)
print("6. 널값 처리: \n", df)

# 구분자가 쉼표가 아닌 csv 파일
df = pd.read_csv(".\data\msft_piped.txt", sep="|", encoding="utf-8")
print("7. 파이프 연산자로 구분된 데이터: \n", df)

# 저장하기
df1 = pd.DataFrame({"deptno":[10, 20, 30, 40],
                   "dname":['R&D','HR','Sales','Management'],
                   "loc":['서울','부산','제주','광주']})
df2 = pd.DataFrame({"emptno":['A001', 'A002', 'A003', 'A004', 'A005'],
                    "salary":[1000, 2000, 3000, 4000, 5000],
                    "hiredate":pd.date_range("2021/02/04", periods=5),
                    "deptno":[10,20,30,20,10]})
# df2.to_csv("./data/chinshin5513.csv")
# df2.to_csv("./data/chinshin5513_2.csv", columns= ["emptno", "salary"], index=False)