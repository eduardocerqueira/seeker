#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고 df.iloc

df = pd.DataFrame(data={'이름': ['홍길동', '이순신', '유관순', '강감찬'],
                        '국어': [10, 45, 22, 45],
                        '수학': [60, 25, 43, 76],
                        "영어":[10, 20, 30, 40],
                        "과학":[10, 20, 30, 40]},
                   index = np.arange(1,5))
print(df)

df.pop("과학")  # 컬럼 삭제 IN PLACE=True
print(df)

# 다중 컬럼 삭제: df.drop(columns = [col1, col2]) IN PLACE=False
df_delete = df.drop(columns = ["수학", "영어"])
print(df)
print(df_delete)  # 복사본을 만들어 준다.(In place = False)

# 다중 컬럼 삭제(값만 쓰기): df.drop([], axis = 1)
df = df.drop(["수학", "영어"], axis = 1)
print(df)