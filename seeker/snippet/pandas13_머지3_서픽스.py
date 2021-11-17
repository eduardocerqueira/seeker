#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고
import seaborn as sns

'''
'''
# 공통 칼럼 병합
df = pd.DataFrame({"x1":["A","B","C"],
                   "y1":[1, 2, 3]})
df2 = pd.DataFrame({"x1":["A","B","C"],
                   "y1":[10, 20, 30]})

m_df = pd.merge(df, df2, on="x1", how="inner", suffixes=["_left","_right"])
print("1. 공통컬럼 병합, inner 병합 \n", m_df)
'''
1. 공통컬럼 병합, inner 병합 
   x1  y1_left  y1_right
0  A        1        10
1  B        2        20
2  C        3        30
'''
