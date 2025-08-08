#date: 2025-08-08T16:51:04Z
#url: https://api.github.com/gists/e6f944d4f1f60c4ac73d32d0b8ddf7f0
#owner: https://api.github.com/users/datavudeja

df = pd.DataFrame({  
'Date': ['2015-05-08', '2015-05-07', '2015-05-06', '2015-05-05', '2015-05-08', '2015-05-07', '2015-05-06', '2015-05-05'],   
'Sym': ['aapl', 'aapl', 'aapl', 'aapl', 'aaww', 'aaww', 'aaww', 'aaww'],  
'Data2': [11, 8, 10, 15, 110, 60, 100, 40],  
'Data3': [5, 8, 6, 1, 50, 100, 60, 120]})  
  
df['Data4'] = df['Data3'].groupby(df['Date']).transform('sum')
df
# Out[74]:
#    Data2  Data3        Date   Sym  Data4
# 0     11      5  2015-05-08  aapl     55
# 1      8      8  2015-05-07  aapl    108
# 2     10      6  2015-05-06  aapl     66
# 3     15      1  2015-05-05  aapl    121
# 4    110     50  2015-05-08  aaww     55
# 5     60    100  2015-05-07  aaww    108
# 6    100     60  2015-05-06  aaww     66
# 7     40    120  2015-05-05  aaww    121