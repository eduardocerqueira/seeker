#date: 2025-01-28T16:53:24Z
#url: https://api.github.com/gists/a6de6bebc1168d8d0603dafd6ebc1e3c
#owner: https://api.github.com/users/TLaconde

import pandas as pd
import matplotlib.pyplot as plt

#Opening data
path = 'Tmax.csv'

df =   p.d. read _csv (path, parse_dates = [ "DATE" ])
df = df.set_index ( "DATE" )

#Chart
plt. plot (df. TMAX )