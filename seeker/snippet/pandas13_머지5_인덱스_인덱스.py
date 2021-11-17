#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고
import seaborn as sns

# 공통 칼럼 병합
df1 = pd.DataFrame({"x1":['a','b','a','a','b','c'],
                   "value":range(1, 7)},
                   index=list("ABDCEF"))
df2 = pd.DataFrame({"group_value":[4,23,5]},
                   index=["A","B","Z"])

print(df1)
m_df = pd.merge(df1, df2, left_index=True, right_index=True, how="inner")  # df1은 컬럼, df2는 인덱스로 조인
m_df = pd.merge(df1, df2, left_index=True, right_index=True, how="outer")  # df1은 컬럼, df2는 인덱스로 조인

# 2. left_on = df1.index, right_index=True
m_df = pd.merge(df1, df2, left_on=df1.index, right_index=True, how="outer")  #

# 2. left_on = df1.index, right_index=True
m_df = pd.merge(df1, df2, left_index=True, right_on=df2.index, how="outer")  #
