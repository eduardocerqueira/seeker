#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고


df = pd.DataFrame(data={'col1': [5, 3, 1, 5, np.nan],
                        'col2': [10, 5, 1, 24, 22],
                        'col3': [5, 2, 1, 4, 8]},
                  index=list("ABCDE"))
print(df)

# 1. unique한 값 반환하기: df[칼럼명].unique()
print("1. col1 칼럼의 unique값:", df['col1'].unique())  # 1. col1 칼럼의 unique값: [ 5.  3.  1. nan]

# 2. unique한 값 갯수 반환하기: df[칼럼명].nunique()  /  Nan은 제외
print("2. col1 칼럼의 unique한 값 갯수 반환:", df['col1'].nunique())  # 2. col1 칼럼의 unique한 값 갯수 반환: 3
print("2. col1 칼럼의 unique한 값 갯수 반환(Nan 포함):", df['col1'].nunique(dropna=False))  # 2. col1 칼럼의 unique한 값 갯수 반환: 4

# 3. unique한 값 빈도 반환하기: df[칼럼명].value_counts()
print("3. col1 칼럼의 unique한 값 빈도 반환하기:", df['col1'].value_counts()) # dropna 가능
'''
5.0    2
3.0    1
1.0    1
'''
# 4. 컬럼명 변경하기: df.rename(columns={딕셔너리} )
copy_df = df.rename(columns={'col1':'c1','col2':'c2','col3':'c3'})
print("4. 컬럼명 변경:", copy_df)

# 4. 인덱스명 변경하기: df.rename(index={딕셔너리} )
copy_df = df.rename(index={'A':'a','B':'b','C':'c'})
print("5. 인덱스명 변경:", copy_df)

df = pd.DataFrame(data={'col1': [5, 3, 1, 5, np.nan],
                        'col2': [10, 5, 1, 24, 22],
                        'col3': [5, 2, 1, 4, 8]},
                  index=list("ABCDE"))

# "6. 지정된 범위 포함 여부(bool): df['컬럼명'].between()  bool반환 → bool색인)
print("6. 지정된 범위 포함 여부(bool): \n", df['col1'].between(1, 4)) # inclusive가 기본(SQL Between A AND B)

# "7. 지정된 범위 포함 여부(bool): df['컬럼명'].isin()  bool반환 → bool색인)
print("7. 지정된 값 포함 여부(bool): \n", df['col1'].isin([1]))  # (SQL in 연산자)

df = pd.DataFrame(data={'col1': [6, 3, 1, 0, np.nan],
                        'col2': [10, 5, 1, 24, 22],
                        'col3': [5, 2, 1, 4, 8]},
                  index=list("ABCDE"))
# "8. 하나라도 참이냐: any(), 모두 참이냐: all()"
print("8. 특정 컬럼값이 모두 참이냐? (bool): \n", df['col1'].all())
print("8. 특정 컬럼값이 하나라도 참이냐? (bool): \n", df['col1'].any())
print("8. 모든 컬럼값이 모두 참이냐? (bool): \n", df.all())

# col1이 하나라도 짝수가 있냐
even_series = df['col1'] % 2 == 0
print("응용 1: \n", even_series.any())

# col1과 col2가 모두 짝수가 있냐?
even_series = df[['col1', 'col2']] % 2 == 0
print("응용 2: \n", even_series.any())
print("응용 2: \n", (df[['col1', 'col2']] % 2 == 0).any())


