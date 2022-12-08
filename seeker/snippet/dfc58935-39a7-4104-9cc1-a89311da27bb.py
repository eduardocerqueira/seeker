#date: 2022-12-08T17:10:14Z
#url: https://api.github.com/gists/d1d7a37ba3e04d7029b2a6abc96fc49e
#owner: https://api.github.com/users/christopherDT

import matplotlib.pyplot as plt

# plot CHW Flow per Deg F. exterior temperature
cmap = {1: 'firebrick', 0: 'gray'}
p = five_min_data.plot(x='Outside Air Temperature', xlabel='Temperature',
                   y='Chilled Water Supply Flow', ylabel='CHW Supply Flow',
                   kind='scatter',
                   title = 'CHW Supply by Outside Temp and Operation',
                   c=[cmap.get(c, 'black') for c in five_min_data.operation],
                   figsize=(10, 6))
