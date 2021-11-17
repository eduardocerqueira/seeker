#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고 감

# 1. dict 이용한 DataFrame 생성
df = pd.DataFrame(data={'col1': [5, 3, 2],
                        'col2': [10, 45, 22],
                        'col3': [6, 2, 43]})  # 클래스를 생성하고 생성자함수를 사용, 딕셔너리를 사용, 네임드 파라미터
# "All arrays must be of the same length"
print(df)  # 데이터 프레임이 생성
print(type(df))  # <class 'pandas.core.frame.DataFrame'>
'''
   col1  col2  col3   시리즈
0     5    10     6
1     3    45     2
2     2    22    43
인덱스
'''
