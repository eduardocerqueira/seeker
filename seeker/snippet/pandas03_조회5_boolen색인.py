#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고 감

df = pd.DataFrame([[1,2,3],[4,5,6],[7,8,9],[9, 10, 11]],
                  index=['cobra', 'viper','sidewinder','sorrento'],
                  columns= ['max_speed', 'shield', 'power'])
print(df)
'''
            max_speed  shield  power
cobra               1       2      3
viper               4       5      6
sidewinder          7       8      9
sorrento            9      10     11
'''

print("1. A 인덱스 라벨 boolean \n", df.loc[[False, False, True, False]])
'''
1. A 인덱스 라벨 boolean 
             max_speed  shield  power
sidewinder          7       8      9
'''
print("",df.loc[df['shield'] > 6] )  #
'''
             max_speed  shield  power
sidewinder          7       8      9
sorrento            9      10     11
'''
print(df['shield'] > 6)  # 조건을 주는 방법
'''
cobra         False
viper         False
sidewinder     True
sorrento       True
Name: shield, dtype: bool
'''
print("1. 인덱스 라벨 boolean, max_speed 컬럼 \n", df.loc[[False, False, True, False], ["max_speed","power"]])
'''
1. 인덱스 라벨 boolean, max_speed 컬럼 
             max_speed  power
sidewinder          7      9
'''

# 6. 논리연산자: &, |, ~
print(df.loc[(df['shield'] >= 6) | (df['max_speed'] > 4)])
'''
            max_speed  shield  power
sidewinder          7       8      9
sorrento            9      10     11
'''
print(df.loc[(df['shield'] >= 6) | (df['max_speed'] > 4), ["shield",'max_speed']])
'''
            shield  max_speed
sidewinder       8          7
sorrento        10          9
'''

print("1. 행렬 모두 boolean 값으로 선택 가능 \n", df.loc[[False, False, True, True],[True, True, True]])
'''
1. 행렬 모두 boolean 값으로 선택 가능 
             max_speed  shield  power
sidewinder          7       8      9
sorrento            9      10     11
'''

print("2. A 인덱스 라벨은 그냥, 컬럼레벨 boolean \n", df.loc[["cobra","viper"], [True, True, False]])
'''
        max_speed  shield
cobra          1       2
viper          4       5
'''

# print("2. A 인덱스 라벨은 그냥, 컬럼레벨 boolean \n", df.loc[["cobra", "viper"], df['max_speed'] >= 3])

print(df.loc[df["max_speed"] > 4,  df.loc['viper'] >= 1])
'''
            max_speed  shield  power
sidewinder          7       8      9
sorrento            9      10     11
'''