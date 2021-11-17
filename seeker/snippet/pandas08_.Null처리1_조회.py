#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고 df.iloc

df = pd.DataFrame(data={'이름': ['홍길동', '이순신', '유관순', '강감찬'],
                        '국어': [10, 45, np.nan, 45],
                        '수학': [60, 25, 43, np.nan],
                        "영어": [10, 20, np.nan, 40],
                        "과학": [10, 20, 30, 40]},
                   index=np.arange(1, 5))
print(df)
'''
    이름    국어    수학    영어  과학
1  홍길동  10.0  60.0  10.0  10
2  이순신  45.0  25.0  20.0  20
3  유관순   NaN  43.0   NaN  30
4  강감찬  45.0   NaN  40.0  40
'''

# 데이터프레임에서 NaN, nan, null, None 찾기
# 1. pd.isna()
print("1. df에 Nan 조회:  \n", pd.isna(df))
print("1. 과학 컬럼에 Nan 조회:  \n", pd.isna(df["과학"]))
print("1. 국어, 과학 컬럼에 Nan 조회:  \n", pd.isna(df[["국어", "과학"]]))

# 2. pd.isnull() → boolean 반환
print("2. df에 Nan 조회:  \n", pd.isnull(df))
print("1. 과학 컬럼에 Nan 조회:  \n", pd.isnull(df["과학"]))
print("1. 국어, 과학 컬럼에 Nan 조회:  \n", pd.isnull(df[["국어", "과학"]]))

# 3. pd.notnull(). ~pd.isnull(), ~pd.isna() → boolean 반환
print("1. df에 Nan 조회:  \n", ~pd.isna(df))
print("1. df에 Nan 조회:  \n", ~pd.isnull(df))
print("1. df에 Nan 조회:  \n", pd.notnull(df))
# 널은 제거하든가, 변환을 시켜줘야 함
