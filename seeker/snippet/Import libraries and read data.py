#date: 2022-05-19T17:29:48Z
#url: https://api.github.com/gists/d06065bedf5e321d269b9adfad321520
#owner: https://api.github.com/users/insightsbees

import pandas as pd
import numpy as np
from raceplotly.plots import barplot

df=pd.read_csv(r'C:\Users\13525\Desktop\Insights Bees\Raceplotly\Data\fortune500.csv')
df.replace('-', np.nan, inplace=True)
df['Revenue']=df['Revenue'].astype(float)
df=df.sort_values(by='Year')