#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고
import seaborn as sns

'''
그룹핑
1. SQL의 group by 기능과 동일
2. df.groupby(by="컬럼명").그룹함수
3. df.groupby(by="컬럼명").agg(사용자함수, 또는 빌트인)
'''
#
# import seaborn as sns
#
# df = sns.load_dataset("mpg")
# print(df.head())
#
# # print(df['origin'].value_counts())
# g_df = df.groupby(by='origin')  # 주소값 출력
# g_df = df.groupby(by='origin')["cylinders"].sum()
#
# print(g_df)

df1 = pd.DataFrame({"deptno":[10, 20, 30, 40],
                   "dname":['R&D','HR','Sales','Management'],
                   "loc":['서울','부산','제주','광주']})

df2 = pd.DataFrame({"emptno":['A001', 'A002', 'A003', 'A004', 'A005'],
                    "salary":[1000, 2000, 3000, 4000, 5000],
                    "hiredate":pd.date_range("2021/02/04", periods=5),
                    "deptno":[10,20,30,20,10]})

print(df1, "\n", df2)
'''
  deptno       dname loc
0      10         R&D  서울
1      20          HR  부산
2      30       Sales  제주
3      40  Management  광주 

   emptno  salary   hiredate  deptno
0   A001    1000 2021-02-04      10
1   A002    2000 2021-02-05      20
2   A003    3000 2021-02-06      30
3   A004    4000 2021-02-07      20
4   A005    5000 2021-02-08      10
'''

# 부서별 월급 합계 및 평균 구하기
sum_df = df2.groupby(by='deptno')["salary"].sum()  # 그룹별 합계
mean_df = df2.groupby(by='deptno')["salary"].mean()  # 그룹별 평균
max_df = df2.groupby(by='deptno')["salary"].max()  # 그룹별 최댓값
min_df = df2.groupby(by='deptno')["salary"].min()  # 그룹별 최솟값
size_df = df2.groupby(by='deptno')["salary"].min()  # 그룹별 갯수


