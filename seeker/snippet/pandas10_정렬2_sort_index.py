#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고 df.iloc

'''
sort_values: 컬럼 값으로 정렬
sort_index: 인덱스 값으로 정렬
'''

import seaborn as sns

df = pd.DataFrame(np.arange(12).reshape(3, 4),
                  index = ["C","A","B"],
                  columns = ['d','a','b','c'])
print(df)

# 인덱스 정렬: df.sort_index( axis = 0 /1)
# axis = 0이면 행 정렬, axis = 1이면 열 정렬
copy_df = df.sort_index(axis=0)
print(copy_df)
'''
   d  a   b   c
A  4  5   6   7
B  8  9  10  11
C  0  1   2   3
'''
copy_df = df.sort_index(axis=1)
print(copy_df)
'''
   a   b   c  d
C  1   2   3  0
A  5   6   7  4
B  9  10  11  8
'''