#date: 2025-08-08T17:12:37Z
#url: https://api.github.com/gists/79bf8ecaa668c4a369c7c43077afb998
#owner: https://api.github.com/users/datavudeja

# Column names
df.columns.values

# column rename
df.rename(columns={"uname":"username"}, inplace=True)

# select columns
df.loc[['row_2','row_3']] 

# drop columns
df.drop(columns=['B', 'C'])

