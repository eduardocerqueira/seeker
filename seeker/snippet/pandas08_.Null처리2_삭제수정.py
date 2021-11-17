#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고 df.iloc

df = pd.DataFrame(data={'이름': ['홍길동', '이순신', '유관순', '강감찬', np.nan],
                        '국어': [10, 45, np.nan, 45, np.nan],
                        '수학': [60, 25, 43, np.nan, np.nan],
                        "영어": [10, 20, 30, 40, np.nan],
                        "과학": [10, 20, 30, 40, np.nan],
                        "체육": [np.nan, np.nan, np.nan, np.nan, np.nan]},
                  index=np.arange(1, 6))
print(df)
'''
    이름    국어    수학  영어  과학
1  홍길동  10.0  60.0  10  10
2  이순신  45.0  25.0  20  20
3  유관순   NaN  43.0  30  30
4  강감찬  45.0   NaN  40  40
'''

# 데이터프레임에서 NaN, nan, null, None 찾아서 삭제하기
# 1. pd.dropna(axis = 0 / 1) → boolean 반환
# print("1. 행에서 null이 하나라도 있는 행 모두 삭제: \n")
# drop_df = df.dropna(axis = 0)  #
# print(drop_df)
'''
    이름    국어    수학  영어  과학
1  홍길동  10.0  60.0  10  10
2  이순신  45.0  25.0  20  20
'''

# print("2. 열에서 null이 하나라도 있는 열 모두 삭제: \n")
# drop_df = df.dropna(axis=1)  #
# print(drop_df)

'''
2. 열에서 null이 하나라도 있는 열 모두 삭제: 
    이름  영어  과학
1  홍길동  10  10
2  이순신  20  20
3  유관순  30  30
4  강감찬  40  40
'''
# 3. pd.dropna( how = "all", axis = 0 / 1)
# Nan이 모두 있는 행, 열만 삭제
print("3. 행에서 null이 Nan라도 있는 열 모두 삭제: \n")
drop_df = df.dropna(axis=0, how='all')
print(drop_df)
'''
    이름    국어    수학    영어    과학  체육
1  홍길동  10.0  60.0  10.0  10.0 NaN
2  이순신  45.0  25.0  20.0  20.0 NaN
3  유관순   NaN  43.0  30.0  30.0 NaN
4  강감찬  45.0   NaN  40.0  40.0 NaN
'''

print("4. 행에서 null이 Nan라도 있는 열 모두 삭제: \n")
drop_df = df.dropna(axis=1, how='all')
print(drop_df)
'''
    이름    국어    수학    영어    과학
1  홍길동  10.0  60.0  10.0  10.0
2  이순신  45.0  25.0  20.0  20.0
3  유관순   NaN  43.0  30.0  30.0
4  강감찬  45.0   NaN  40.0  40.0
5  NaN   NaN   NaN   NaN   NaN
'''

# 5. Null 값을 임의의 값으로 변경하기
# df.fillna(변경값):
print("5. 모든 Null값을 N/A로 변경 \n")
fill_na = df.fillna("N/A")
print(fill_na)
'''
    이름    국어    수학    영어    과학   체육
1  홍길동  10.0  60.0  10.0  10.0  N/A
2  이순신  45.0  25.0  20.0  20.0  N/A
3  유관순   N/A  43.0  30.0  30.0  N/A
4  강감찬  45.0   N/A  40.0  40.0  N/A
5  N/A   N/A   N/A   N/A   N/A  N/A
'''

# 6. Null 값을 임의의 값으로 변경하기(칼럼 지정)
# df.fillna({칼럼명:변경값, 칼럼명:변경값}): # 딕셔너리 형태로
print("6. Null값을 칼럼마다 다른 값으로 변경 \n")
fill_na = df.fillna({"국어": 0, "영어": 10, "수학": -1})
print(fill_na)

# ffill(앞에 있는 값을 변경), bfill(뒤에 있는 값을 변경)
# 7. Null 값 주변 앞뒤를 가져와서 변경하기(칼럼 지정)
print("7. Null 값 주변 앞뒤를 가져와서 변경하기 \n")
fill_na = df.fillna(method="ffill")
print(fill_na)
'''
    이름    국어    수학    영어    과학  체육
1  홍길동  10.0  60.0  10.0  10.0 NaN
2  이순신  45.0  25.0  20.0  20.0 NaN
3  유관순  45.0  43.0  30.0  30.0 NaN
4  강감찬  45.0  43.0  40.0  40.0 NaN
5  강감찬  45.0  43.0  40.0  40.0 NaN
'''
print("7. Null 값 주변 앞뒤를 가져와서 변경하기 \n")
fill_na = df.fillna(method="bfill")
print(fill_na)
'''
    이름    국어    수학    영어    과학  체육
1  홍길동  10.0  60.0  10.0  10.0 NaN
2  이순신  45.0  25.0  20.0  20.0 NaN
3  유관순  45.0  43.0  30.0  30.0 NaN
4  강감찬  45.0   NaN  40.0  40.0 NaN
5  NaN   NaN   NaN   NaN   NaN NaN
'''
