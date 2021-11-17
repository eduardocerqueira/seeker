#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고
import seaborn as sns

# 공통 칼럼 병합
df1 = pd.DataFrame({"x1":['a','b','a','a','b','c'],
                   "value":range(1, 7)})
df2 = pd.DataFrame({"group_value":[4,23,5]},
                   index=["a","b","c"])

m_df = pd.merge(df1, df2, left_on="x1", right_index=True, how="inner")  # df1은 컬럼, df2는 인덱스로 조인
print(m_df)
'''
  x1  value  group_value
0  a      1            4
2  a      3            4
3  a      4            4
1  b      2           23
4  b      5           23
5  c      6            5
'''

df2 = pd.DataFrame({"group_value":[4,23,5]},
                   index=["a","b","d"])

m_df = pd.merge(df1, df2, left_on="x1", right_index=True, how="inner")  # df1은 컬럼, df2는 인덱스로 조인
print(m_df)
'''.
  x1  value  group_value
0  a      1            4
2  a      3            4
3  a      4            4
1  b      2           23
4  b      5           23
Inner 조인이라서 한 쪽이 안 나옴
'''

df1 = pd.DataFrame({"x1":['a','b','a','a','b','c'],
                   "value":range(1, 7)})
df2 = pd.DataFrame({"group_value":[4,23,5]},
                   index=["a","b","c"])
m_df = pd.merge(df1, df2, left_on="x1", right_on=df2.index, how="outer")  # 인덱스를 직접 조인,
print(m_df)