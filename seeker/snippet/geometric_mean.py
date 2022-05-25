#date: 2022-05-25T17:09:23Z
#url: https://api.github.com/gists/1091fca1505053711cf9f2c09d12f05b
#owner: https://api.github.com/users/alexlenail

from scipy.stats import gmean
def geometric_mean(df):
    '''https://www.reddit.com/r/learnpython/comments/mq5ea7/pandas_calculate_geometric_mean_while_ignoring/'''
    return df.replace(0, np.nan).apply(lambda row: gmean(row[~row.isna()]), axis=1).fillna(0)
