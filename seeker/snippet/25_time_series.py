#date: 2025-08-08T17:12:37Z
#url: https://api.github.com/gists/79bf8ecaa668c4a369c7c43077afb998
#owner: https://api.github.com/users/datavudeja

# http://pandas.pydata.org/pandas-docs/stable/timeseries.html#timeseries
# easy sample
# resample
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
rng = pd.date_range('1/1/2012', periods=100, freq='S')
ts.resample('5Min').sum()

# time zone
rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
ts = pd.Series(np.random.randn(len(rng)), rng)
ts_utc = ts.tz_localize('UTC')
ts_utc.tz_convert('US/Eastern')

# convert monthly data to quarterly data
rng = pd.date_range('1/1/2012', periods=5, freq='M')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ps = ts.to_period()
ps.to_timestamp()

## full examples

##
# generate timestamp
pd.Timestamp(datetime(2012, 5, 1))
pd.Timestamp('2012-05-01')
pd.Timestamp(2012, 5, 1)

# generate Period (time span)
pd.Period('2011-01')
pd.Period('2012-05', freq='D')

# generate series using timestamp/period
dates = [pd.Timestamp('2012-05-01'), pd.Timestamp('2012-05-02'), pd.Timestamp('2012-05-03')]
ts = pd.Series(np.random.randn(3), dates)

# convert date-like object to timestamp
pd.to_datetime(pd.Series(['Jul 31, 2009', '2010-01-10', None]))
pd.to_datetime(['2005/11/23', '2010.12.31'])
pd.to_datetime(['04-01-2012 10:00'], dayfirst=True)
pd.to_datetime('2010/11/12')
pd.Timestamp('2010/11/12')

# convert pandas columns to timestamp index
df = pd.DataFrame({'year': [2015, 2016],  'month': [2, 3], 'day': [4, 5],'hour': [2, 3]})

# to_timestamp error handling
pd.to_datetime(['2009/07/31', 'asd'], errors='raise')
pd.to_datetime(['2009/07/31', 'asd'], errors='ignore')
pd.to_datetime(['2009/07/31', 'asd'], errors='coerce') # convert errors to NAT


# get range of timestamps
index = pd.date_range('2000-1-1', periods=1000, freq='M')
index = pd.bdate_range('2012-1-1', periods=250) # business day
rng = pd.date_range(datetime(2011, 1, 1), datetime(2012, 1, 1))

pd.Timestamp.min
pd.Timestamp.max
rng = pd.date_range(start, end, freq='BM')

# access range
ts['1/31/2011']
ts[datetime(2011, 12, 25):]
ts['10/31/2011':'12/31/2011']
ts['2011']
ts['2011-6']
dft['2013-1':'2013-2-28 00:00:00']
dft['2013-1-15':'2013-1-15 12:30:00']

# date shift
d + pd.tseries.offsets.DateOffset(months=4, days=5)
from pandas.tseries.offsets import *
d + DateOffset(months=4, days=5)
d - 5 * BDay()
d + BMonthEnd()
offset = BMonthEnd()
offset.rollforward(d)
offset.rollback(d)
d + Week()
d + Week(weekday=4)
d + YearEnd()
d + YearEnd(month=6)

rng = pd.date_range('2012-01-01', '2012-01-03')
s = pd.Series(rng)
rng + DateOffset(months=2)
s + DateOffset(months=2)

s - Day(2)
td = s - pd.Series(pd.date_range('2011-12-29', '2011-12-31'))


# Hoilday
from pandas.tseries.holiday import USFederalHolidayCalendar
bday_us = CustomBusinessDay(calendar=USFederalHolidayCalendar())
dt = datetime(2014, 1, 17)
dt + bday_us

# business hour
bh = BusinessHour()
pd.Timestamp('2014-08-01 10:00') + bh
pd.Timestamp('2014-08-01 10:00') + BusinessHour(2)

# custom business hour
bh = BusinessHour(start='11:00', end=time(20, 0))
pd.Timestamp('2014-08-01 09:00') + bh

# annoted offsets
pd.Timestamp('2014-01-02') + MonthBegin(n=4)
pd.Timestamp('2014-01-31') + MonthEnd(n=1)
pd.Timestamp('2014-01-02') + MonthEnd(n=0)

# shift/lag
ts = ts[:5]
ts.shift(1)
ts.shift(5, freq=offsets.BDay())
ts.shift(5, freq='BM')

# frequency conversion
dr = pd.date_range('1/1/2010', periods=3, freq=3 * offsets.BDay())
ts = pd.Series(randn(3), index=dr)
ts.asfreq(BDay())
ts.asfreq(BDay(), method='pad')

# resampling (e.g. 1min -> 5min)
rng = pd.date_range('1/1/2012', periods=100, freq='S')
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
df.resample('M', on='date')
ts.resample('5Min').sum()
ts.resample('5Min').mean()
ts.resample('5Min').ohlc() # open high low close
ts.resample('5Min', closed='right').mean()
ts.resample('5Min', closed='left').mean()

# offset alias
# B	business day frequency
# D	calendar day frequency
# W	weekly frequency
# M	month end frequency
# BM	business month end frequency
# MS	month start frequency
# BMS	business month start frequency
# Q	quarter end frequency
# A	year end frequency
# H	hourly frequency
# T, min	minutely frequency
# S	secondly frequency

# upsampling (e.g. 5min to min)
ts[:2].resample('250L').ffill()

# sparse resampling
# omitted

################
## Time span
#################

# period
p = pd.Period('2012', freq='A-DEC')
p + 1  #Period('2013', 'A-DEC')
p = pd.Period('2014-07-01 09:00', freq='H')
p + Hour(2)
p + timedelta(minutes=120)

# period range
prng = pd.period_range('1/1/2011', '1/1/2012', freq='M')
pd.PeriodIndex(['2011-1', '2011-2', '2011-3'], freq='M')
pd.PeriodIndex(start='2014-01', freq='3M', periods=4)
ps = pd.Series(np.random.randn(len(prng)), prng)
idx = pd.period_range('2014-07-01 09:00', periods=5, freq='H')

#PeriodIndex Partial String Indexing¶
ps['2011-01']
ps[datetime(2011, 12, 25):]
ps['10/31/2011':'12/31/2011']
ps['2011']
dfp['2013-01-01 10H']
dfp['2013-01-01 10H':'2013-01-01 11H']
p = pd.Period('2011', freq='A-DEC')

# convert period to other freq
p.asfreq('M', how='start')

# time zone
rng_pytz = pd.date_range('3/6/2012 00:00', periods=10, freq='D',tz='Europe/London')
ts = pd.Series(np.random.randn(len(rng)), rng)
ts_utc = ts.tz_localize('UTC')
ts_utc.tz_convert('US/Eastern')
rng_eastern = rng_utc.tz_convert('US/Eastern')
didx.tz_localize(None) # remove timezone

# convert timezone from US/Eastern to Asia/Tokyo
df['timestamp'] = pd.to_datetime(df.DATE) #convert to datetime
df.index = df.timestamp # convert to datetimeindex
ind =df.index
ind = ind.tz_localize('US/Eastern') # convert to tz-aware US time
ind = pd.to_datetime(ind.values)  # convert to UTC without timezone
ind = ind + pd.tseries.offsets.DateOffset(hours=9) # convert to Tokyo time
df.index = ind
df.timestamp = ind

