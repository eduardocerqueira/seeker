#date: 2025-08-08T17:12:37Z
#url: https://api.github.com/gists/79bf8ecaa668c4a369c7c43077afb998
#owner: https://api.github.com/users/datavudeja

# loc, iloc, ix
# loc select by names of columns or rows
df.loc[['row_2','row_3']] # select columns
df.loc[:, ['col_1','col_2']] # select rows

# iloc select by numerical index
df.iloc[[2,3]]
df.iloc[:, [1,2]]

# ix select by numerical index for row and column name
df.ix[[3,7], "row_b"]


# simple conditions
df3 = df2[df2['Time0']==2012]
df3 = df3[df3['Value1'] > 20000]
df3 = df3[df3['Country0'] == df3['Country1']]
df = df[pd.notnull(df['latitude'])] # not null
df[df['category'].str.contains('national')] # string condition

# multiple conditions
df[df['id'].isin(['109', '673'])]
df[(df['category'] == 'national') & (df['is_removed'] == '1')]

# negative conditions
df = df[~df['id'].isin(['1', '2', '3'])]

# by index
df['20130102':'20130104']
df[0:3]

# selected columns
 df[['A','B']]
  
# combine index and column selection
df.loc['20130102':'20130104',['A','B']]
df.loc['20130102','A','B']  # return scalar

# select row close to some value
df.ix[(df.CCC-aValue).abs().argsort()]