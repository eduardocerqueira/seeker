#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고 감

df = pd.DataFrame(data={'col1': [5, 3, 2, 1],
                        'col2': [10, 45, 22, 3],
                        'col3': [6, 2, 43, 4],
                        'col4': [4, 5, 6, 7]},
                  index=list("ABCD"))
print(df)
'''
        0     1     2     3
       col1  col2  col3  col4
0 A     5    10     6     4
1 B     3    45     2     5
2 C     2    22    43     6
3 D     1     3     4     7
'''
print("1. A 인덱스 위치 조회 \n", df.iloc[0])  # 시리즈
'''
1. A 인덱스 라벨 조회 
col1     5
col2    10
col3     6
col4     4
'''

print("1. A C 인덱스 위치 조회 \n", df.iloc[[0, 2]])
'''
1. A C 인덱스 위치 조회 
    col1  col2  col3  col4
A     5    10     6     4
C     2    22    43     6
'''

print("1. A인덱스 col2 인덱스 위치 조회 \n", df.iloc[0, 1])  # 10, scalar
print("1. A인덱스 col2, col4 인덱스 위치 조회 \n", df.iloc[0, [1, 2]])  # 10, scalar
'''
1. A인덱스 col2, col4 인덱스 위치 조회 
 col2    10
col3     6
'''

print("1. A인덱스, C인덱스 /  col3 인덱스 위치 조회 \n", df.iloc[[0, 2], 2])  # 10, scalar
'''
1. A인덱스, C인덱스 /  col3 인덱스 위치 조회 
 A     6
C    43
Name: col3, dtype: int64
'''

print("1. A인덱스, C인덱스 /  col1, col3 인덱스 위치 조회 \n", df.iloc[[0, 2], [0, 2]])  #
'''
1. A인덱스, C인덱스 /  col1, col3 인덱스 위치 조회 
    col1  col3
A     5     6
C     2    43
'''

