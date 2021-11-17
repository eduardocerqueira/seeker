#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고 감

# 1. 하나의 열을 갖는 시리즈 생성 → 프레임 변경
s = pd.Series(["홍길동", "이순신", "유관순"], name="성명")  # 시리즈(하나의 열)
print(s)
s_df = s.to_frame()  # 컬럼이 하나 있는 시리즈 만드는 법
print(s_df)
print(type(s))  # <class 'pandas.core.series.Series'>
print(type(s_df))  # <class 'pandas.core.frame.DataFrame'>

# 1. 여러 개의 열을 갖는 시리즈 생성 → 프레임 변경
name_s = pd.Series(["박지성", "이영표"])
age_s = pd.Series([21, 25])
address_s = pd.Series(["수원", "서울"])

df = pd.DataFrame([name_s, age_s, address_s])
print(df)  # 행단위로 나오는 것을 체크
'''
     0    1
0  박지성  이영표
1   21   25
2   수원   서울
'''
print(df.T)  # 열단위로 나오는 것을 체크
'''
     0   1   2
0  박지성  21  수원
1  이영표  25  서울
'''

df.columns = ["국가대표1", "국가대표2"]
df.index = ["성명","연령","출신"]
print(df)  # 행단위로 나오는 것을 체크
'''
   국가대표1 국가대표2
성명   박지성   이영표
연령    21    25
출신    수원    서울
'''
print(df.T)  # 열단위로 나오는 것을 체크
'''
        성명  연령  출신
국가대표1  박지성  21  수원
국가대표2  이영표  25  서울
'''