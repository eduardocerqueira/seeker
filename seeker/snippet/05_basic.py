#date: 2025-08-08T17:12:37Z
#url: https://api.github.com/gists/79bf8ecaa668c4a369c7c43077afb998
#owner: https://api.github.com/users/datavudeja

# creation
df = pd.DataFrame(data = BabyDataSet, columns=['Names', 'Births']) 
df = pd.read_csv('file.csv')
df.to_csv('name.csv', encoding='utf-8') 

# df info
df.columns.values # DF info
nrow = df.shape[0] # DF info

# filter 
df.loc[:, ['col_1','col_2']] # select rows by string index
df.iloc[[2,3]] # select rows by numerical index. double bracket gives number, instead of series
df.iloc[:, [1,2]]

# update
df.loc[df.AAA >= 5,'BBB'] = -1 #  update AAA: condition, BBB: columns to be changed

# iteration
for idx, row in df.iterrows():
  print([row.value1, row["value2"]])
df.value3 = df.apply(lambda x: x.value1 + x.value2, axis = 1)
grp = iris.groupby('Species').apply(eval('np.sum'))

# manage dataframe
df.reset_index()
df.sort_values(by=[('Group1', 'C')], ascending=False) # multi index sort
df1.dropna(how='any')
df['col'] = df['col'].fillna(-1).astype(int)