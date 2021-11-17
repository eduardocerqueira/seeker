#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고 df.iloc

df = pd.DataFrame(data={'이름': ['홍길동', '이순신', '유관순', '강감찬'],
                        '국어': [10, 45, 22, 45],
                        '수학': [60, 25, 43, 76]},
                  index=np.arange(1, 5))
print(df)

eng_series =pd.Series([90, 80, 70, 60], index = [1, 2, 3, 4])
print(eng_series)  # 시리즈는 0부터, df는 1부터 들어감
# 인덱스 값을 반드시 지정해 주어야 한다.
df["영어"] = eng_series
print(df)
'''
    이름  국어  수학  영어
1  홍길동  10  60  90
2  이순신  45  25  80
3  유관순  22  43  70
4  강감찬  45  76  60
'''
