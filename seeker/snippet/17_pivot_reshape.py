#date: 2025-08-08T17:12:37Z
#url: https://api.github.com/gists/79bf8ecaa668c4a369c7c43077afb998
#owner: https://api.github.com/users/datavudeja

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pivot.html
long_df = pd.DataFrame({'vname': ['one', 'one', 'one', 'two', 'two','two','one', 'one', 'one', 'two', 'two','two'],
                   'country': ['A', 'B', 'C', 'A', 'B', 'C','A', 'B', 'C', 'A', 'B', 'C'],
                   'year': ['x1', 'x1', 'x1', 'x1', 'x1', 'x1','x2', 'x2', 'x2', 'x2', 'x2', 'x2'],
                   'val': [1, 2, 3, 4, 5, 6,7, 8, 9, 10, 11, 12]})

# reshape from long to wide
wide_df = long_df.pivot(index=['country','year'], columns='vname', values="val").reset_index()

# reshape from wide to long (neither "vname" or "val" doesn't exist in wide_df. final DF contains [country, year, vname, val] coluumns)
new_long_df = wide_df.melt(id_vars=['country','year'], var_name='vname', value_name="val")
