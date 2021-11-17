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

n=np.arange(0,2, 0.2)

plt.subplot(4, 1, 1)
plt.plot(n, n, "bo", label = "a")
plt.subplot(4, 1, 2)
plt.plot(n, n**2, linestyle="--", color="red", label = "b")
plt.subplot(4, 1, 3)
plt.plot(n, n**3, label = "c")
plt.xlabel("x축")
plt.ylabel("y축")
plt.legend()
plt.show()
