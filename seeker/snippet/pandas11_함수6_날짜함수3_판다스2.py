#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고
import seaborn as sns

# 2. 판다스 날짜함수 정리
# 지정된 범위의 날짜 생성 반환:
copy_date = pd.date_range("2002/01/01", "2002/01/08")
print(copy_date)
'''
DatetimeIndex(['2002-01-01', '2002-01-02', '2002-01-03', '2002-01-04',
               '2002-01-05', '2002-01-06', '2002-01-07', '2002-01-08'],
              dtype='datetime64[ns]', freq='D')
'''
copy_date = pd.date_range(start="2002/01/01", periods=5, freq="M")
print(copy_date)  # 월의 마지막 날을 출력함
'''
DatetimeIndex(['2002-01-31', '2002-02-28', '2002-03-31', '2002-04-30', '2002-05-31'],
               dtype='datetime64[ns]', freq='M')
'''

copy_date = pd.date_range(start="2002/01/01", periods=5, freq="2M")
print(copy_date)  # 월의 마지막 날을 출력함
'''
DatetimeIndex(['2002-01-31', '2002-03-31', '2002-05-31', '2002-07-31',
               '2002-09-30'],
              dtype='datetime64[ns]', freq='2M')
'''

copy_date = pd.date_range(start="2002/01/01", periods=5, freq="Y")
print(copy_date)  # 연의 마지막 날을 출력함
'''
DatetimeIndex(['2002-12-31', '2003-12-31', '2004-12-31', '2005-12-31',
               '2006-12-31'],
              dtype='datetime64[ns]', freq='A-DEC')
'''

# 활용 → 연 매출 출력
date10 = pd.date_range(end="2021/07/27", periods=10, freq = "Y")
print(date10)
df = pd.DataFrame({"매출":np.arange(1,11)},
                  index = date10)
print(df)
