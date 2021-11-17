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

x = np.arange(4)
y = [100, 300, 400, 100]
plt.bar(x,y)
plt.xticks(x, ["2017", "2018", "2019", "2020"])
plt.show()