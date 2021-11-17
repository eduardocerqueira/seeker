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
import pandas as pd

plt.rc("font", family="Malgun Gothic")
'''
박스플롯(Box plot)
- 4분위수 지정 가능
'''
np.random.seed(1234)
df = pd.DataFrame(np.random.random(10))  # 벡터
print(df)
plt.boxplot(df)
file = plt.gcf()  # 저장할 준비
plt.show()
file.savefig("test.png")  # 저장하기
# 색상 지정, 데이터 포맷 지정, 뽑기 등도 가능함
