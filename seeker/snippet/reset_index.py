#date: 2023-01-19T16:48:13Z
#url: https://api.github.com/gists/680bc612b4c5716147b114288c9d263a
#owner: https://api.github.com/users/JohannesHechler

# prep
import pandas as pd
data = [['Alex',10],['Bob',12],['Clarke',13]]
df = pd.DataFrame(data, columns=['Name','Age'])

# without reset_index(), the grouping column becomes the index in the new dataframe
df.groupby('Name').agg({'Age':sum})

# ... with it, it remains a column to work with
df.groupby('Name').agg({'Age':sum}).reset_index()
