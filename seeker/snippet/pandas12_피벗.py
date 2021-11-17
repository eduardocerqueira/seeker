#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고
import seaborn as sns

'''
pd.pivot(): 행을 열로, long을 wide로
'''
df= pd.DataFrame({"date":[2000,2000,2000,2001,2001,2001,2002,2002,2002],
                  "item":["A","B","C","A","B","C","A","B","C"],
                  "value":list(range(1,10))})
print(df)

copy_df = df.pivot(index="date",
                   columns="item",
                   values="value")
print(copy_df)

copy_df = df.pivot(index="item",
                   columns="date",
                   values="value")
print(copy_df)
