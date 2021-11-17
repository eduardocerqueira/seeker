#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고 감

# 1. dict 이용한 DataFrame 생성
df = pd.DataFrame({'col1': {2000: 300, 2001: 150},
                   'col2': {2000: 400, 2001: 100},
                   'col3': {2000: 500, 2001: 80, 2002: 180}})
print(df)
'''
       col1   col2  col3
2000  300.0  400.0   500
2001  150.0  100.0    80
2002    NaN    NaN   180  딕셔너리가 중첩되면, 내부 딕셔너리의 key값이 인덱스로 들어옴
'''

print(df.T)  # 전치도 가능
'''
       2000   2001   2002
col1  300.0  150.0    NaN
col2  400.0  100.0    NaN
col3  500.0   80.0  180.0
'''