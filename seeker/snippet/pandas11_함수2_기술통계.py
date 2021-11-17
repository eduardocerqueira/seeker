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

print("1. df의 행의 갯수: \n", len(df))  # 5
print("2. 칼럼별(열) 최댓값: \n", df.max())
print("2. 인덱스별(행) 최댓값: \n", df.max(axis=1))
print("3. 칼럼별(열) 최댓값의 인덱스: \n", df.idxmax(axis=0))
print("3. 인덱스별(행) 최댓값: \n", df.idxmax(axis=1))
print("4. 칼럼별(열) 최솟값의 인덱스: \n", df.idxmin(axis=0))
print("4. 칼럼별(열) 최솟값의 인덱스: \n", df.idxmin(axis=1))

print("5. 칼럼별(열) 합계: \n", df.sum(axis=0))
print("5. 인덱스별(행) 합계: \n", df.sum(axis=1))

print("6. 칼럼별(열) 평균: \n", df.mean(axis=0))
print("6. 인덱스별(행) 평균: \n", df.mean(axis=1))

print("7. 칼럼별(열) 중앙값: \n", df.median(axis=0))
print("7. 인덱스별(행) 중앙값: \n", df.median(axis=1))

print("8. 칼럼별(열) 사분위수: \n", df.quantile([0.25, 0.75], axis=0))
print("8. 인덱스별(행) 사분위수: \n", df.quantile([0.25, 0.75], axis=1))

print("9. 칼럼별(열) 분산: \n", df.var(axis=0))
print("9. 인덱스별(행) 분산: \n", df.var(axis=1))

print("9. 칼럼별(열) 표준편차: \n", df.std(axis=0))
print("9. 인덱스별(행) 표준편차: \n", df.std(axis=1))

print("10. 칼럼별(열) 데이터 숫자(Nan 제외): \n", df.count(axis=0))
print("10. 인덱스별(행) 데이터 숫자(Nan 제외): \n", df.count(axis=1))


print("11. 칼럼별(열) 누적합: \n", df.cumsum(axis=0))
print("11. 인덱스별(행) 누적합: \n", df.cumsum(axis=1))

print("12. 칼럼별(열) 누적곱: \n", df.cumprod(axis=0))
print("12. 인덱스별(행) 누적곱: \n", df.cumprod(axis=1))

print("12. 칼럼별(열) 누적최댓값: \n", df.cummax(axis=0))
print("12. 인덱스별(행) 누적최댓값: \n", df.cummax(axis=1))

print("13. 칼럼별(열) 누적최솟값: \n", df.cummin(axis=0))  # 열로 진행하면서 최솟값을 비교
print("13. 인덱스별(행) 누적최솟값: \n", df.cummin(axis=1))  # 행으로 진행하면서 최솟값을 비교

print("14. 통계데이터 통합: \n", df.describe())  # 데이터 통계 통합, SQl의 desc
'''
14. 통계데이터 통합: 
            col1       col2      col3
count  4.000000   5.000000  5.000000
mean   2.750000  12.400000  4.000000
std    1.707825  10.212737  2.738613
min    1.000000   1.000000  1.000000
25%    1.750000   5.000000  2.000000
50%    2.500000  10.000000  4.000000
75%    3.500000  22.000000  5.000000
max    5.000000  24.000000  8.000000
'''