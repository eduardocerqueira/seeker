#date: 2025-08-08T17:12:37Z
#url: https://api.github.com/gists/79bf8ecaa668c4a369c7c43077afb998
#owner: https://api.github.com/users/datavudeja

# single value
df.at[dates[0],'A'] = 0

# column
df.loc[:,'D'] = np.array([5] * len(df))
df['logic'] = np.where(df['AAA'] > 5,'high','low');
# column rename
df.rename(columns={"uname":"username"}, inplace=True)


# update information
df.loc[df.AAA >= 5,'BBB'] = -1 # AAA: condition, BBB: columns to be changed
df.ix[[3,7],'BBB'] = -1 # use ix, not iloc


# group by update 1. update only variable which satisfies group conditions
moras = moras.reset_index()
tmp = moras.groupby(['wid'])["mid"].idxmax()
moras.ix[tmp,"pitch_change"] = np.nan

# group by update. Assign unique ide within group
df['domainId'] = df.groupby('orgid')['orgid'].rank(method='first')


#----------
# deleteing
#----------
# column
del df[name] # column

# row
df.drop(df.index[[1,3]], inplace=True)
df = df[(df.a == 2)]
df = df[~np.isnan(df['corr'])] # delete NaN
