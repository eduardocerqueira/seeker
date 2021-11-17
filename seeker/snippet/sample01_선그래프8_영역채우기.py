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

plt.rc("font", family="Malgun Gothic")

'''
마커

'''
x = [1, 2, 3, 4]
y = [1, 4, 9, 16]
plt.plot(x, y)  # y축으로 됨, x값도 함꼐 생성
plt.xlabel("X축")
plt.ylabel("Y축")
plt.axis([0, 5, 0, 20])  # 축 범위 지정
plt.fill_between(x[1:3], y[1:3], alpha=0.3)  # 리스트의 슬라이싱이므로 인덱스 값을 나타냄, 투명도
plt.show()
