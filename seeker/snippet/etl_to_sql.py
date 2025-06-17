#date: 2025-06-17T17:04:04Z
#url: https://api.github.com/gists/3635cf9f72657846dae2ee52fecd1011
#owner: https://api.github.com/users/alexmatiasas

import pandas as pd
from sqlalchemy import create_engine

# 1) Loading example data
df = pd.read_csv("matches_sample.csv")

# 2) Connection to DB
engine = create_engine("postgresql://user:pass@host/dbname")

# 3) Load and query
df.to_sql("matches", engine, if_exists="replace", index=False)
result = engine.execute("SELECT team, COUNT(*) AS games FROM matches GROUP BY team")
print(result.fetchall())