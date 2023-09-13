#date: 2023-09-13T17:06:24Z
#url: https://api.github.com/gists/ba20c3f466ace639a27d26364e0612bc
#owner: https://api.github.com/users/xavier211192

import pandas as pd

file = 'flights.parquet'
df = pd.read_parquet(path=file, engine='pyarrow')
print(df.head(100))