#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고
import seaborn as sns

df = sns.load_dataset("mpg")

# apply함수 용도: 사용자가 작성한 함수를 한번에 데이터프레임의 행과 열에 적용가능한 함수

# 1. df.apply(함수 [, 옵션])
print(df.iloc[[0], :])

def fun1(n):
    return n[0]  # 첫 번째 인덱스를 반환하는 함수

# df = df.apply(fun1)  # 기본적으로는 컬럼별로 적용됨(axis = 0)
# print(df)
# df = df.apply(lambda n:n[0], axis = 0 ) # 람다 함수도 가능
# print(df)

copy_df = df.apply(lambda n:n[0:3], axis = 0)
print("3. 첫번째 행~ 세번째 행 까지 반환하기(apply) \n", copy_df)

df = pd.DataFrame({'국어': [10, 45, 22, 45],
                   '수학': [60, 25, 43, 76]},
                   index=np.arange(1, 5))
print("4. 컬럼별 총합: \n", df.apply(sum))
print("4. 행별 총합: \n", df.apply(sum, axis = 1))
df["총합"] = df.apply(sum, axis = 1)
print(df)
copy_df = df.append(df.apply(sum), ignore_index=True)
copy_df.index = [1, 2, 3, 4, "합계"]
print(copy_df)
copy_df = df.append(df.apply(np.mean), ignore_index=True)
copy_df.index = [1, 2, 3, 4, "평균"]
print(copy_df)
#---------------------------------------------------
# Series에도 가능함
print("5. 특정 칼럼에서 apply 적용", copy_df["총합"] > 30)
print("5. 특정 칼럼에서 apply 적용", copy_df["총합"].apply(lambda n: n > 30))  # 시리즈로 가능함
