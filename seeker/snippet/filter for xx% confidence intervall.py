#date: 2022-03-03T17:01:05Z
#url: https://api.github.com/gists/bda1a8d0c55024c9f8ba19b8befe7d70
#owner: https://api.github.com/users/TGreenGH

def exclude_outliers(df, column, confidence_intervall):
    lower = (100-confidence_intervall)/2
    upper = 100 - lower
    
    maxlimit = np.percentile(df[column], [lower, upper])[1]
    minlimit = np.percentile(df[column], [lower, upper])[0]
    
    ind_drop = df[(df[column] > maxlimit) | (df[column] < minlimit)].index

    df_ex = df.drop(ind_drop, inplace = False)
    
    return maxlimit, minlimit, df_ex