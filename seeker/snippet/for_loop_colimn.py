#date: 2022-04-13T16:42:19Z
#url: https://api.github.com/gists/b386fb4ab1d9cc0e972ddc0bc540b09c
#owner: https://api.github.com/users/harish-maddukuri

import pandas as pd, numpy as np
df = pd.read_cas(Market_data.csv, header=None)
market = []
for  row  in  df [ 'Country' ]:
     if  row  ==  'Germany' :
         market.append ( 'DACH' )
     elif  row  ==  'Austria' :
         market.append ( 'DACH' )
     elif  row  ==  'Uk' :
         market.append ( 'UK / IE' )
     elif  row  ==  'Ireland' :
         market.append( 'UK / IE' )
     else :
         market. append ( 'Other' )
df [ 'Market' ] =  market
print(df)


"""
#Output
   Account_Number        City     Country Account_value
2            5364   Stuttgart     Germany         12000
3             224   Nuremberg     Germany          4000
4            7653      Vienna     Austria         57000
5           65447     Insberg     Austria         43000
6           76589      Berlin     Germany        157000
7           76543      Berlin     Germany          1200
8          785578    Salzburg     Austria          2000
9            6658      Dublin     Ireland          5100
10            765        Cork     Ireland          6200
11          23457      London          UK         70000
12           4459      Galway     Ireland         31000
13            987      London          UK          4300
14          80865    Edinburg          UK         36200
15          45673  Manchester          UK          5700
16         643456   Liverpool          UK          4300
17          65436  Dusseldorf     Germany         16000
18           4326     Cologne     Germany         11008
19          53211    Duisburg     Germany          1400
20         112235      Vienna     Austria         67000
21        6547674      London          UK         59300
22           6648     Bristol          UK          1206
23        4346787        Roma       Italy            24
24         445567        Kyiv     Ukraine         23000
25           5765  Bratislava  Slovacchia         45500
26            778      Athens      Greece          2100
27          34588      Madrid       Spain          6700
28         808490  Barcellona       Spain         38000
"""