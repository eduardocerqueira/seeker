#date: 2025-08-08T17:12:37Z
#url: https://api.github.com/gists/79bf8ecaa668c4a369c7c43077afb998
#owner: https://api.github.com/users/datavudeja

# http://sinhrks.hatenablog.com/entry/2014/10/13/005327

grp = iris.groupby('Species')
grp['len'].sum() #single column
grp.sum()                 # all column
grp[['width', 'len']].sum() # multipl columns

# custom functions. Custom function's input is series. 
grp[ 'Length'].apply(np.sum)
grp[ 'Length'].apply(lambda x: ",".join(x))

# multiple custom functions
grp.agg({'Length': [np.sum, np.mean], 'Width': [np.sum, np.mean]})

# add new column based on groupby. 
iris["avgren"] = grp["Length"].transform("sum")

# Complex cumulation
# Using quarterly frequency y-y growth rate, create level data.
df = pd.DataFrame(['country_id','quarter','year', 'yoygrowth'])
df = df.sort_values(["country_id", "quarter", "year"])
def cumulative_func(df):
    results = []
    for group in df.groupby(["country_id", "quarter"]).indices.values():
        level = 1
        result = []
        for val in df.iloc[group].yoygrowth.values:

            level *= 1 + 0.01 * val
            result.append(level)
        results.append(pd.Series(result, index=group))
    # return pd.concat(results)
    return pd.concat(results).reindex(df.index)

df = df.reset_index(drop=True)
df["level"] = cumulative_func(df)
