#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고
import seaborn as sns

# 2. 판다스 날짜함수 정리

# 활용 → 연 매출 출력
date10 = pd.date_range(end="2021/07/27", periods=10, freq = "D")
print(date10)
df = pd.DataFrame({"날짜": date10})
# print(df)

# 연도, 월, 일을 뽀는 방법
# print(dir(df["날짜"].dt))
'''
print(dir(df["날짜"].dt))
'ceil', 'date', 'day', 'day_name', 'day_of_week', 'day_of_year', 'dayofweek', 'dayofyear', 
'days_in_month', 'daysinmonth', 'floor', 'freq', 'hour', 'is_leap_year', 'is_month_end', 
'is_month_start', 'is_quarter_end', 'is_quarter_start', 'is_year_end', 'is_year_start', 'isocalendar', 
'microsecond', 'minute', 'month', 'month_name', 'nanosecond', 'normalize', 'quarter', 'round', 'second', 
'strftime', 'time', 'timetz', 'to_period', 'to_pydatetime', 'tz', 'tz_convert', 'tz_localize', 'week', 
'weekday', 'weekofyear', 'year']
'''
print("1. 연도만 출력", df["날짜"].dt.year)
print("2. 월만 출력", df["날짜"].dt.month)
print("2. 월만 출력", df["날짜"].dt.month_name())
print("3. 일만 출력", df["날짜"].dt.day)
print("3. 일만 출력", df["날짜"].dt.day_name())  # 연월일의 영어이름 출력
print("4. 주 출력", df["날짜"].isocalendar().week)  # 몇 번째 주?

#######
# 날짜를 문자로 변경
print("5. 날짜를 문자로", df["날짜"].astype(str))  # 몇 번째 주?
