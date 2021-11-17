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
print("1. A 인덱스 라벨 조회 \n", df.loc["A"])
'''
1. A 인덱스 라벨 조회 
col1     5
col2    10
col3     6
col4     4
'''
print("1. A,C 인덱스 라벨 조회 \n", df.loc[["A", "C"]])  # 행라벨끼리는 대괄호 2개
'''
    col1  col2  col3  col4
A     5    10     6     4
C     2    22    43     6
'''
print("2. A 인덱스 col2 컬럼 라벨 조회 \n", df.loc["A", "col2"])  # 행렬 함께 쓸 때는 괄호 하나, 스칼라로 반환 10
print("2. A 인덱스 col2,col3 컬럼 라벨 조회 \n", df.loc["A", ["col2", "col3"]])  # 시리즈로 반환
'''
2. A 인덱스 col2,col3 컬럼 라벨 조회 
 col2    10
col3     6
'''

print("3. A,C 인덱스 col2 컬럼 라벨 조회 \n", df.loc[["A","C"],"col2"])  # 시리즈로 반환
'''
3. A,C 인덱스 col2 컬럼 라벨 조회 
 A    10
C    22
'''

print("4. A,C 인덱스 col2, col3 컬럼 라벨 조회 \n", df.loc[["A","C"],["col2","col3"]])  # 데이터 프레임으로 반환
'''
4. A,C 인덱스 col2, col3 컬럼 라벨 조회 
    col2  col3
A    10     6
C    22    43
'''

