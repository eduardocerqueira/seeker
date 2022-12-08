#date: 2022-12-08T17:10:12Z
#url: https://api.github.com/gists/47b0efe91a2f4f1be51cf983a5c67cee
#owner: https://api.github.com/users/christopherDT

def get_workweek(df):
    if df['weekday'] in [0, 1, 2, 3, 4]:
        return 'week'
    else:
        return 'weekend'

# day type feature coding
five_min_data['hour'] = five_min_data.index.hour
five_min_data['weekday'] = five_min_data.index.weekday
five_min_data['workweek'] = five_min_data.apply(get_workweek, axis=1)

cmap = {'week': 'firebrick', 'weekend': 'gray'} # plot flow per hour, color by weekday vs weekend

p = five_min_data.plot(x='hour',
                   y='Chilled Water Supply Flow',
                   ylabel='CHW Supply Flow',
                   kind='scatter',
                   title='CHW Flow per Hour',
                   c=[cmap.get(c, 'black') for c in five_min_data.workweek])
