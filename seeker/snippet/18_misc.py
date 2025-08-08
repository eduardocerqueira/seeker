#date: 2025-08-08T17:12:37Z
#url: https://api.github.com/gists/79bf8ecaa668c4a369c7c43077afb998
#owner: https://api.github.com/users/datavudeja

# Iteration
for idx, row in df.iterrows():
  print([row.value1, row["value2"]])
 
df.value3 = df.apply(lambda x: x.value1 + x.value2, axis = 1)

# reset index
df.reset_index(level=None, drop=False, inplace=False, col_level=0, col_fill='')
#http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.reset_index.html

df.sort_values(by=[('Group1', 'C')], ascending=False) # multi index sort

#na
df1.dropna(how='any')
df1.fillna(value=5)
pd.isnull(df1)
df1.[df1.x.notna()]

#histgram
s.value_counts()

