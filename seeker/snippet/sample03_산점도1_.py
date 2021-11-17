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
'''
산점도(scatter plot)
- 두 개의 값을 x와 y값의 좌표를 점으로 표현하는 그래프.
'''

x= np.random.rand(10000)
y= np.random.rand(10000)
# colors = np.random.rand(50)
# s_size = (30* np.random.rand(50))**2
plt.scatter(x, y) # 마커의 크기
# plt.scatter(x, y, c=colors, s=s_size) # 마커의 크기
plt.show()
# random.seed를 통해 랜덤 값 고정 가능함
# 마커의 모양, 크기 변경 가능