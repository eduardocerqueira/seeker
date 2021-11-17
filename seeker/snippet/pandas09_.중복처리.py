#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고 df.iloc

df = pd.DataFrame({'k1': ['one'] * 3 + ['two'] * 4,
                   'k2': [1, 1, 2, 3, 3, 4, 4]})
print(df)
'''
    k1  k2
0  one   1
1  one   1
2  one   2
3  two   3
4  two   3
5  two   4
6  two   4
'''
# 중복행 조회 df.duplicated()
print("1. DataFrame에 중복행 조회")
print(df.duplicated())  # Series
print(df.duplicated(keep="first"))  # Series, 처음에 발견된 데이터를 기준으로 중복 체크
'''
1. DataFrame에 중복행 조회
0    False
1     True
2    False
3    False
4     True
5    False
6     True
dtype: bool
'''

print(df.duplicated(keep="last"))  # 마지막에 발견된 데이터를 기준으로 중복체크

# 2. 특정 칼럼 중복행 조회: df.duplicated()
print("2. 특정 칼럼 중복행 조회")
print(df.duplicated("k1"))  # print(df.duplicated("k1", "k2"))
'''
1. DataFrame에 중복행 조회
0    False
1     True
2     True
3    False
4     True
5     True
6     True
dtype: bool
'''

# 3. 중복행 제거 df.drop_duplicates(), 중복 데이터를 제거 후 데이터프레임 반환
print("3. 칼럼 중복행 제거")
drop_df = df.drop_duplicates()
print(drop_df)  # ignore_index=True로 인덱스 재정렬 가능
'''
3. 특정 칼럼 중복행 조회
    k1  k2
0  one   1
2  one   2
3  two   3
5  two   4
'''