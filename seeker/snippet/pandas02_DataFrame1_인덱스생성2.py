#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고 감

'''
Data Frame 인덱스 생성
1. 기본 인덱스 → 자동으로 0, 1, 2 지정
2. 명시적 인덱스 → df.index = [1, 2, 3, 4]
3. 기존 컬럼을 인덱스로 변경 → df.set_index()  복사본을 만들어 줌
    사용하고 있는 데이터에 덮어쓰고 싶다면, In Place 
    기존 인덱스를 덮어써서 제거됨, 기존 인덱스는 없어짐
4. 기존 인덱스는 컬럼으로 변경하고, 새로운 인덱스 생성
    df.reset_index  복사본을 만들어 줌
    df.reset_index(inplace=True) 사용하고 있는 데이터에 덮어쓰고 싶다면, 
5. 기존 인덱스를 삭제하고 새로운 인덱스 생성
    df.reset_index(drop=True) → 복사본 반환
    df.reset_index(drop=True, inplace=True) → 사용하고 있는 데이터에 덮어쓰고 싶다면, 
6. df.reindex: 기존 인덱스의 재배치
    ex) 'A','B','C' → 'C','B','A'
7. ignore_index = True (df 병합 시, index가 중복됨)
    
    
'''
df = pd.DataFrame(data={'col1': [5, 3, 2],
                        'col2': [10, 45, 22],
                        'col3': [6, 2, 43]},
                  index=['B', 'C', 'A'])
print("1. 기본 인덱스 \n", df)
'''
    col1  col2  col3
B     5    10     6
C     3    45     2
A     2    22    43
'''
print("--------------------------------------------------")
copy_df = df.reindex(['A', 'B', 'C', 'D'])  # 기존에 없던 인덱스는 NaN으로 생성됨
print("6. 인덱스 재배치 \n", copy_df)
'''
    col1  col2  col3
A   2.0  22.0  43.0
B   5.0  10.0   6.0
C   3.0  45.0   2.0
D   NaN   NaN   NaN
'''
print("--------------------------------------------------")
df = pd.DataFrame(data={'col1': [5, 3, 2]})
df2 = pd.DataFrame(data={'col1': [50, 30, 20]})
merge_df = pd.concat([df, df2])
print("7. 인덱스 중복 제거(에러) \n", merge_df)
'''
    col1
0     5
1     3
2     2
0    50
1    30
2    20
'''
merge_df = pd.concat([df, df2], ignore_index=True)
print("3. 인덱스 중복 제거 \n", merge_df)
'''
3. 인덱스 중복 제거 
    col1
0     5
1     3
2     2
3    50
4    30
5    20

'''