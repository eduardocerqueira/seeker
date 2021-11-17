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
print("1. B 인덱스 라벨 ~ 끝까지 조회 \n", df.loc["B":])
'''
1. B 인덱스 라벨 ~ 끝까지 조회 
    col1  col2  col3  col4
B     3    45     2     5
C     2    22    43     6
D     1     3     4     7
'''

print("1. B 인덱스 라벨 ~ C까지 조회 \n", df.loc["B":"C"])  # 범위 모두 inclusive
'''
1. B 인덱스 라벨 ~ C까지 조회 
    col1  col2  col3  col4
B     3    45     2     5
C     2    22    43     6
'''

print("1. B 인덱스 라벨부터 끝까지, col2 컬럼 조회 \n", df.loc["B":, "col2"])  # 범위 모두 inclusive
'''
1. B 인덱스 라벨부터 끝까지, col2 컬럼 조회 
 B    45
C    22
D     3
'''

print("1. 인덱스 처음부터 C까지, col2, col1 컬럼 조회 \n", df.loc[:"C", ["col2", "col1"]])  # 범위 모두 inclusive
'''
1. 인덱스 처음부터 C까지, col2, col1 컬럼 조회 
    col2  col1
A    10     5
B    45     3
C    22     2
'''

print("1. 인덱스 B, col2~ 끝까지 \n", df.loc["B", "col2":])  # 범위 모두 inclusive
'''
1. 인덱스 B, col2~ 끝까지 
 col2    45
col3     2
col4     5
Name: B, dtype: int64
'''

print("6. 인덱스 A, B, 칼럼 처음부터 ~ col2까지 \n", df.loc[["A", "B"], :"col2"])  # 범위 모두 inclusive
'''
6. 인덱스 A, B, 칼럼 처음부터 ~ col2까지 
    col1  col2
A     5    10
B     3    45
'''

print("7. 인덱스 처음부터 C까지, 칼럼 처음부터 ~ col2까지 \n", df.loc[:"C", :"col2"])  # 범위 모두 inclusive
'''
7. 인덱스 처음부터 C까지, 칼럼 처음부터 ~ col2까지 
    col1  col2
A     5    10
B     3    45
C     2    22
'''