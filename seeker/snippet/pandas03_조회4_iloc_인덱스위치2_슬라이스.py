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
# 인덱스 라벨 슬라이싱
print("1. B 인덱스 위치 조회 \n", df.iloc[1:])
print("2. B~C 인덱스 위치 조회 \n", df.iloc[1:2])  # exclusive
print("3. 인덱스 위치 처음부터_C까지, col2, col1 조회 \n", df.iloc[:3, [1, 0]])  # exclusive
'''
3. 인덱스 위치 처음부터_C까지, col2, col1 조회 
    col2  col1
A    10     5
B    45     3
C    22     2
'''

print("4. 인덱스 위치 처음부터_C까지, col2, col1 조회 \n", df.iloc[:3, [1, 0]])  # exclusive

print("5. B 인덱스 위치, col2 ~ 끝 조회 \n", df.iloc[1, 1:])  # exclusive

print("6. A,B 인덱스 위치, 처음부터 col2까지 조회 \n", df.iloc[[0,1], :2])  # exclusive

print("7. 인덱스 처음부터 C까지,칼럼 처음부터 col2까지 조회 \n", df.iloc[:3, :2])  # exclusive
'''
7. 인덱스 처음부터 C까지,칼럼 처음부터 col2까지 조회 
    col1  col2
A     5    10
B     3    45
C     2    22
'''