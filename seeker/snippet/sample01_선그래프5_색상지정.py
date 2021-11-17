#date: 2021-11-17T16:51:31Z
#url: https://api.github.com/gists/723bb65f3577fc45f6ca0e7e9f7bb4d9
#owner: https://api.github.com/users/chinshin5513


'''
  시각화
  1. matplotlib 패키지 사용
  2. pandas 시각화 제공 ( matplotlib 기능 일부분 제공 )
  3. matplotlib  +seaborn 패키지
  설치
  pip install matplotlib
'''
import matplotlib.pyplot as plt

# 한글설정
plt.rc("font", family="Malgun Gothic")

'''
   색상 지정
   1) 'b','r' ....
   2) 색상명
   3) 16진수 색상
'''

plt.plot([1,2,3,4],[1,4,9,16], color="#e35f62")
plt.xlabel("X값")
plt.ylabel("Y값")
plt.axis([0,5,0,20])
plt.show()

