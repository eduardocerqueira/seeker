#date: 2022-12-08T17:10:13Z
#url: https://api.github.com/gists/38c7fc6223657a96cdfe91e0e544b278
#owner: https://api.github.com/users/christopherDT

def get_opp_status(df):

    if (df['workweek'] == 'week') & ((df['hour'] >= 10) | (df['hour'] == 0)):
        return 1
    else:
        return 0
    
five_min_data['operation'] = five_min_data.apply(get_opp_status, axis=1)
