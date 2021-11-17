#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고 df.iloc

df = pd.DataFrame(data={'col1': [5, 3, 2, 1],
                        'col2': [10, 45, 22, 3],
                        'col3': [6, 2, 43, 4],
                        'col4': [4, 5, 6, 7]},
                  index=list("ABCD"))
# print(df)
'''
   col1  col2  col3  col4
A     5    10     6     4
B     3    45     2     5
C     2    22    43     6
D     1     3     4     7
'''

# 1. loc(라벨)로 조회 후 값 변경
# A행, col2를 찾아서 10을 100으로 바꾼다.
print(df.loc["A", "col2"])  # 스칼라
df.loc["A", "col2"] = 100
# print(df)
'''
   col1  col2  col3  col4
A     5   100     6     4
B     3    45     2     5
C     2    22    43     6
D     1     3     4     7
'''

# B행, D행의
print(df.loc[["B","D"],["col1","col2"]])
df.loc[["B","D"],["col1","col2"]] = -1
print(df)

# 2. iloc[위치]로 찾기
print(df.iloc[[2, 3], [0, 1]])
df.iloc[[2, 3], [0, 1]] = 1000
print(df)
'''
   col1  col2  col3  col4
A     5   100     6     4
B    -1    -1     2     5
C  1000  1000    43     6
D  1000  1000     4     7
'''

# 역방향 바꾸는 것도 가능
df.iloc[[2, 3], [-2, -1]] = 200
print(df)
'''
   col1  col2  col3  col4
A     5   100     6     4
B    -1    -1     2     5
C  1000  1000   200   200
D  1000  1000   200   200
'''

