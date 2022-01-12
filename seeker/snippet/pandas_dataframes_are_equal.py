#date: 2022-01-12T17:06:17Z
#url: https://api.github.com/gists/273070c2d42cbc6ced2ddd723babb049
#owner: https://api.github.com/users/ttor

def pandas_dataframes_are_equal(df1, df2):
    return list(sorted(df1.columns)) == list(sorted(df2.columns)) and \
           len(df1.merge(df2, how="outer", indicator=True).query("_merge!='both'"))==0