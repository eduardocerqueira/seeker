#date: 2025-08-08T17:12:37Z
#url: https://api.github.com/gists/79bf8ecaa668c4a369c7c43077afb998
#owner: https://api.github.com/users/datavudeja

#change column type
df['name'] = df['name'].astype('str')
df['col'] = df['col'].fillna(-1).astype(int)
df['col'] = pd.to_numeric(df['col'])