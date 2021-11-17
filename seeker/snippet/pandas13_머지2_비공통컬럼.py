#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고
import seaborn as sns

# 비공통 칼럼 병합
df = pd.DataFrame({"x1": ["A","B","C"],
                   "x2": [1, 2, 3]})
df2 = pd.DataFrame({"y1": ["A","B","C"],
                   "y2": [10, 20, 30],
                    "y3": [100, 200, 300]})
m_df = pd.merge(df, df2, left_on="x1", right_on="y1", how="inner")  # 아우터 조인(df)
print("1. 비공통컬럼 병합, inner 병합 \n", m_df)
'''
1. 비공통컬럼 병합, inner 병합 
   x1  x2 y1  y2   y3
0  A   1  A  10  100
1  B   2  B  20  200
2  C   3  C  30  300
'''

m_df = pd.merge(df, df2, left_on="x1", right_on="y1", how="inner")\
    .drop(columns=['y1'])\
    .rename(columns={"x1":"x1/y1"})  # 아우터 조인(df)
print("1. 비공통컬럼 병합, inner 병합 \n", m_df)
'''
1. 비공통컬럼 병합, outer 병합
   x1/y1  x2  y2   y3
0     A   1  10  100
1     B   2  20  200
2     C   3  30  300
'''

# 비공통 칼럼 병합
df = pd.DataFrame({"x1": ["A","B","D"],
                   "x2": [1, 2, 3]})
df2 = pd.DataFrame({"y1": ["A","B","C"],
                   "y2": [10, 20, 30],
                    "y3": [100, 200, 300]})
# m_df = pd.merge(df, df2, left_on="x1", right_on="y1", how="left")  # 아우터 조인(df)
# m_df = pd.merge(df, df2, left_on="x1", right_on="y1", how="right")  # 아우터 조인(df)
m_df = pd.merge(df, df2, left_on="x1", right_on="y1", how="outer")  # 아우터 조인(df)
print("3. 비공통컬럼 병합, outer 병합 \n", m_df)
'''
3. 비공통컬럼 병합, outer 병합 
     x1   x2   y1    y2     y3
0    A  1.0    A  10.0  100.0
1    B  2.0    B  20.0  200.0
2    D  3.0  NaN   NaN    NaN
3  NaN  NaN    C  30.0  300.0
'''