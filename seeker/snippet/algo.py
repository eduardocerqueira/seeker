#date: 2024-01-12T16:59:45Z
#url: https://api.github.com/gists/256d376df5d8602cb5014c7376376388
#owner: https://api.github.com/users/devmnj

# I encounter proplem while using Historical data
# For yfinance most of the case the data is in good shape
# The first thing I am looking for index column, of numbers. if it exist, need to replace with datetime
# the last thing I want make sure that the date column is in panda datetime type otherwise have to covert it. 

df =pd.DataFrame(data['Success'])
df =df[['datetime','open','high','low','close','volume']]
df['datetime']= pd.to_datetime(df['datetime']) 
df.set_index('datetime', inplace=True)