#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고 df.iloc

df = pd.DataFrame(data={'이름': ['홍길동', '이순신', '유관순', '강감찬'],
                        '국어': [10, 45, 22, 45],
                        '수학': [60, 25, 43, 76],
                        "영어": [10, 20, 30, 40],
                        "과학": [10, 20, 30, 40]},
                   index=np.arange(1, 5))
print(df)

df2 = pd.DataFrame(data={'이름': ['송하나', '송병구'],
                         '국어': [10, 45],
                         '수학': [60, 25],
                         "영어": [10, 20],
                         "과학": [10, 20]},
                   index=np.arange(1, 3))
append_df = df.append(df2, ignore_index=True)
print(append_df)
'''
    이름  국어  수학  영어  과학
0  홍길동  10  60  10  10
1  이순신  45  25  20  20
2  유관순  22  43  30  30
3  강감찬  45  76  40  40
4  송하나  10  60  10  10
5  송병구  45  25  20  20
'''

# 2. pd.concat([df1, df2], axis = 0)
concat_df = pd.concat([df, df2], axis=0, ignore_index=True)
print(concat_df)