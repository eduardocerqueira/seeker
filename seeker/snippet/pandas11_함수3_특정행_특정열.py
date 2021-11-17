#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고


df = pd.DataFrame(data={'col1': [5, 3, 1, 2, np.nan],
                        'col2': [10, 5, 1, 24, 22],
                        'col3': [5, 2, 1, 4, 8]},
                  index=list("ABCDE"))
print(df)

print("1. 특정 컬럼의 최댓값:", df['col1'].max())  # 시리즈를 메서드하므로 스칼라
print("2. 특정 컬럼의 최댓값 위치:", df['col1'].idxmax())  # 시리즈를 메서드하므로 스칼라
print("3. 특정 컬럼의 최솟값:", df['col1'].min())  # 시리즈를 메서드하므로 스칼라
print("4. 특정 컬럼의 최솟값 위치:", df['col1'].idxmin())  # 시리즈를 메서드하므로 스칼라
print("5. 특정 컬럼의 합계:", df['col1'].sum())  # 시리즈를 메서드하므로 스칼라
print("6. 특정 컬럼의 평균:", df['col1'].mean())  # 시리즈를 메서드하므로 스칼라
print("7. 특정 컬럼의 중앙값:", df['col1'].median())  # 시리즈를 메서드하므로 스칼라
print("8. 특정 컬럼의 갯수:", df['col1'].count())  # 시리즈를 메서드하므로 스칼라
print("9. 특정 컬럼의 누적합:", df['col1'].cumsum())  # 시리즈를 메서드하므로 스칼라
print("10. 특정 컬럼의 누적곱:", df['col1'].cumprod())  # 시리즈를 메서드하므로 스칼라

# 컬럼(열) 단위
print("9. 다중 컬럼의 최댓값:", df[['col1', 'col2']].max())  # 시리즈를 메서드하므로 스칼라

# 행 단위(iloc)
print("10. 특정행의 최댓값:", df.loc['A'].max())  # 시리즈를 메서드하므로 스칼라
print("11. 특정행의 최솟값:", df.loc['A'].min())  # 시리즈를 메서드하므로 스칼라
print("12. 특정행의 합계:", df.loc['A'].sum())  # 시리즈를 메서드하므로 스칼라
print("13. 특정행의 평균:", df.loc['A'].mean())  # 시리즈를 메서드하므로 스칼라
print("14. 특정행의 중앙값:", df.loc['A'].median())  # 시리즈를 메서드하므로 스칼라
print("15. 특정행의 갯수:", df.loc['A'].count())  # 시리즈를 메서드하므로 스칼라
print("16. 특정행의 누적 합:", df.loc['A'].cumsum())  # 시리즈를 메서드하므로 스칼라
print("17. 특정행의 누적 곱:", df.loc['A'].cumprod())  # 시리즈를 메서드하므로 스칼라