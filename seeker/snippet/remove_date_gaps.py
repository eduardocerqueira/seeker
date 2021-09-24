#date: 2021-09-24T17:13:18Z
#url: https://api.github.com/gists/cd72d21b0635b2607336cd998a88dbac
#owner: https://api.github.com/users/zjwarnes

from datetime import datetime
dt_all = pd.date_range(start=df.index.min(),end=df.index.max(), freq='1min') # Get all dates
dt_obs = [d.strftime("%Y-%m-%d %H:%M:%S") for d in pd.to_datetime(df.index)] # retrieve the dates that ARE in the original datset
dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d %H:%M:%S").tolist() if not d in dt_obs] # define dates with missing values

fig.update_layout(
    xaxis={
        'rangebreaks':[dict(values=dt_breaks)]
    }
)