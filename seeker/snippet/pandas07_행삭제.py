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
# 다중 행삭제 df.drop(index = [idx1, idx2]), 삭제된 새로운 프레임을 반환
drop_df = df.drop(index=[1, 2])
print(drop_df)

# 다중 행삭제
drop_df = df.drop([1, 3], axis=0)  # 인덱스를 1부터 센다는 점
print(drop_df)