#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고 감

# 1. dict 이용한 DataFrame 생성
df = pd.DataFrame(data= {'col1':[5, 3, 2],
                         'col2':[10, 45, 22],
                         'col3':[6, 2, 43]},
                         index=[1, 2, 3])
print(df)

df = pd.DataFrame(data= {'col1':[5, 3, 2],
                         'col2':[10, 45, 22],
                         'col3':[6, 2, 43]},
                         index=['A', 'B', 'C'])  # 인덱스도 변경가능
print(df)
df.columns = ['c1','c2','c3']  # 컬럼 변경하기
print(df)
