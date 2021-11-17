#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고 감

'''
Data Frame 인덱스 생성
1. 기본 인덱스 → 자동으로 0, 1, 2 지정
2. 명시적 인덱스 → df.index = [1, 2, 3, 4]
3. 기존 컬럼을 인덱스로 변경 → df.set_index()  복사본을 만들어 줌
    사용하고 있는 데이터에 덮어쓰고 싶다면, In Place 
    기존 인덱스를 덮어써서 제거됨, 기존 인덱스는 없어짐
4. 기존 인덱스는 컬럼으로 변경하고, 새로운 인덱스 생성
    df.reset_index  복사본을 만들어 줌
    df.reset_index(inplace=True) 사용하고 있는 데이터에 덮어쓰고 싶다면, 
5. 기존 인덱스를 삭제하고 새로운 인덱스 생성
    df.reset_index(drop=True) → 복사본 반환
    df.reset_index(drop=True, inplace=True) → 사용하고 있는 데이터에 덮어쓰고 싶다면, 
    
'''
df = pd.DataFrame(data={'col1': [5, 3, 2],
                        'col2': [10, 45, 22],
                        'col3': [6, 2, 43]})
print("1. 기본 인덱스 \n", df)
print("--------------------------------------------------")
df = pd.DataFrame(data={'col1': [5, 3, 2],
                        'col2': [10, 45, 22],
                        'col3': [6, 2, 43]},
                  index=[1, 2, 3])
print("2. 명시적 인덱스 \n", df)
print("--------------------------------------------------")
df = pd.DataFrame(data={'col1': [5, 3, 2],
                        'col2': [10, 45, 22],
                        'col3': [6, 2, 43],
                        'keys': ['AA', 'BB', 'CC']})
# copy_df = df.set_index(['keys'])  # 복사본이 만들어져야 함
df.set_index(['keys'], inplace=True)  # 덮어쓰기
print("3. 기존 칼렴을 인덱스로 변경 \n", df)
print("--------------------------------------------------")
df.reset_index(inplace=True)
print("4. 기존 인덱스는 컬럼으로, 새로운 인덱스는 인덱스로 변경 \n", df)
print("--------------------------------------------------")
df.reset_index(inplace=True, drop=True)
print("5. 기존 인덱스는 제거하고, 새로운 인덱스는 인덱스로 만들 때 \n", df)

