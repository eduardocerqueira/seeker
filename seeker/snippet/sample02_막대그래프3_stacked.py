#date: 2021-11-17T16:51:31Z
#url: https://api.github.com/gists/723bb65f3577fc45f6ca0e7e9f7bb4d9
#owner: https://api.github.com/users/chinshin5513

'''
시각화
1. matplotlib 패키지 사용
2. pandas 시각화 제공(matplotlib 기능)
3. matplotlib + seaborn 패키지
'''
import matplotlib.pyplot as plt
import numpy as np

plt.rc("font", family="Malgun Gothic")
men = [20, 25, 30, 25, 27]
women = [25, 32, 34, 20, 25]

x= np.arange(5)
bar1 = plt.bar(x, men)
bar2 = plt.bar(x, women, bottom=men)

plt.title("학년별 성별 비율")
plt.xticks(x, ["1학년", "2학년", "3학년", "4학년", "5학년"])
plt.legend()
plt.show()