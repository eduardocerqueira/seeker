#date: 2021-11-08T17:00:23Z
#url: https://api.github.com/gists/f9cb100ee82d4635b6888ea4c0bc24a2
#owner: https://api.github.com/users/jamiegl

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

scatter_matrix(daily_bike_share_raw[features + [label]], figsize=[12, 8])
plt.show()