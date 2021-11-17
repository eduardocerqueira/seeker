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
파이차트(pie plot)
- 두 개의 값을 x와 y값의 좌표를 점으로 표현하는 그래프.
'''
ratio = [1, 2, 1, 2]
label = ["Apple", "Banana", "Melon", "Mandarin"]
color = ["GreenYellow", "MediumPurple", "Navy", "Pink"]
plt.pie(ratio, labels=label, autopct="%.1f%%", colors=color, explode=[0.1, 0.1, 0.1, 0.1])  # 데이터 포맷을 잘 확인할 것
file = plt.gcf()  # 파일에 저장할 준비
plt.show()
file.savefig("test.png")
# 색상 지정, 데이터 포맷 지정, 뽑기 등도 가능함
