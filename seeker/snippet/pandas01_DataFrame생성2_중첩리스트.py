#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고 감

# 1.중첩 리스트를 이용한 DataFrame 생성
df = pd.DataFrame([[5, 3, 2],[10, 45, 22],[6, 2, 43]])
# print(df)
'''
    0   1   2  컬럼 인덱스
0   5   3   2
1  10  45  22
2   6   2  43
행 인덱스
'''
# 컬럼명, 인덱스명을 지정하는 방법
df = pd.DataFrame([[5, 3, 2],[10, 45, 22],[6, 2, 43]],
                  index=(list('ABC')),
                  columns=['col1', 'col2', 'col3'])
print(df)
'''
   col1  col2  col3
A     5     3     2
B    10    45    22
C     6     2    43
'''

# 3.다차원 행렬을 이용하는 것도 가능

