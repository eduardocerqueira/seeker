#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고 감

'''

'''
df = pd.DataFrame(data={'col1': [5, 3, 2],
                        'col2': [10, 45, 22],
                        'col3': [6, 2, 43]})
print(df)
print("1. col1 컬럼 조회[] : \n", df['col1'])
'''
1. col1 컬럼 조회[] : 
 0    5
1    3
2    2
'''
print("1. col1 컬럼 조회 . : \n", df.col1)
print("2. 여러 컬럼 조회[] : \n", df[['col1', 'col2']])
'''
2. 여러 컬럼 조회[] : 
    col1  col2
0     5    10
1     3    45
2     2    22
'''

