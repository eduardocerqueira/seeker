#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고
import seaborn as sns

'''
merge
1. SQL의 조인과 비슷함
2. INNER merge(일치하는 애들만 나옴, 일치 안하면 누락), OUTER merge(누락된 것 포함)

pd.merge(df1, df2, 옵션) 속성, 이너 조인 등등
 how 이너 조인, 아우터 조인
 on 공통컬럼
 index 공통 인덱스
3. 컬럼x - 컬럼x 연결(공통 컬럼) (공통컬럼:  on 속성, 비공통컬럼: left_on, right_on)
4. 컬럼x - 컬럼y 연결(비공통 컬럼)
5. 컬럼x -인덱스0 연결
6. 인덱스0 - 인덱스a 연결
'''
# 공통 칼럼 병합
df = pd.DataFrame({"x1":["A","B","C"],
                   "y1":[1, 2, 3]})
df2 = pd.DataFrame({"x1":["A","B","C"],
                   "x2":[10, 20, 30],
                    "x3":[100, 200, 300]})

m_df = pd.merge(df, df2, on="x1", how="inner")
print("1. 공통컬럼 병합, inner 병합 \n", m_df)
m_df = pd.merge(df, df2[["x1","x3"]], on="x1", how="inner")
print("1. 공통컬럼 병합, 특정컬럼만 참여하기 \n", m_df)

# ___________________________________________________________
df = pd.DataFrame({"x1":["A","B","C"],
                   "y1":[1, 2, 3]})
df2 = pd.DataFrame({"x1":["A","B","D"],
                   "x2":[10, 20, 30],
                    "x3":[100, 200, 300]})
m_df = pd.merge(df, df2, on="x1", how="left") # 아우터 조인(df)
print("3. 공통컬럼 병합, outter 병합 \n", m_df)
'''
3. 공통컬럼 병합, outer 병합 
   x1  y1    x2     x3
0  A   1  10.0  100.0
1  B   2  20.0  200.0
2  C   3   NaN    NaN
'''
m_df = pd.merge(df, df2, on="x1", how="right") # 아우터 조인(df2)
print("3. 공통컬럼 병합, outer 병합 \n", m_df)
'''
3. 공통컬럼 병합, outer 병합 
   x1   y1  x2   x3
0  A  1.0  10  100
1  B  2.0  20  200
2  D  NaN  30  300
'''

m_df = pd.merge(df, df2, on="x1", how="outer") # 아우터 조인(df2)
print("3. 공통컬럼 병합, outer 병합 \n", m_df)
'''
3. 공통컬럼 병합, outer 병합 
   x1   y1    x2     x3
0  A  1.0  10.0  100.0
1  B  2.0  20.0  200.0 
2  C  3.0   NaN    NaN
3  D  NaN  30.0  300.0
'''
m_df = pd.merge(df, df2, on="x1", how="outer", indicator=True) # 아우터 조인(df2)
print("3. 공통컬럼 병합, outer 병합, 병합 어떻게 됐는지 알려주기 \n", m_df)

# 필터링(병합된 결과에서 query하기: SQL)
m_df = pd.merge(df, df2, on="x1", how="outer").query("x1 in ['A','B']")  # 쿼리하는 방법(조건 걸고)
print("3. 공통컬럼 병합, 쿼리 \n", m_df)

m_df = pd.merge(df, df2, on="x1", how="outer", indicator=True).query("_merge == 'left_only'")  # 아우터 조인(df2)
print("3. 공통컬럼 병합, 쿼리. 병합된 종류 \n", m_df)

m_df = pd.merge(df, df2, on="x1", how="outer", indicator=True).query("x3 > 150") # 아우터 조인(df2)
print("3. 공통컬럼 병합, 쿼리. 병합된 종류 \n", m_df)

m_df = pd.merge(df, df2, on="x1", how="outer", indicator=True).query("_merge == 'left_only'").drop(columns=["_merge"]) # 아우터 조인(df2)
print("3. 공통컬럼 병합, 쿼리. 병합된 종류 \n", m_df)

