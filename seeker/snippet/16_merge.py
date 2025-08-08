#date: 2025-08-08T17:12:37Z
#url: https://api.github.com/gists/79bf8ecaa668c4a369c7c43077afb998
#owner: https://api.github.com/users/datavudeja

#http://pandas.pydata.org/pandas-docs/stable/merging.html

#--------------
#Concat
#---------------
#horizontal concat
frames = [df1, df2, df3]
result = pd.concat(frames)

# vertical concat, if index is not meaningfu, better ignore
result = df1.append(df2, ignore_index=True)
result = df1.append([df2, df3], ignore_index=True)

pd.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False,
          keys=None, levels=None, names=None, verify_integrity=False,
          copy=True)

#  Add row
result = df1.append(one_row, ignore_index=True)

# Join
result = pd.merge(left, right, how='left', on=['key1', 'key2'])

# how: =left/right/outer/inner