#date: 2022-06-16T16:57:52Z
#url: https://api.github.com/gists/5f13689d09e5926528f81e9fa075c26b
#owner: https://api.github.com/users/edersonbadeca

# coding: utf-8

# This sample code shows how to create a timeseries based on date periods to calculate
# the frequency that an event occurs in a hour.
# Links:
# - https://pandas.pydata.org/docs/reference/api/pandas.date_range.html
# - https://pandas.pydata.org/docs/getting_started/intro_tutorials/09_timeseries.html?highlight=resample#resample-a-time-series-to-another-frequency

import pandas as pd
import numpy as np
rng = pd.date_range('1/1/2011', periods=5, freq='60S')
ts = pd.Series(np.random.randint(0, 5, len(rng)), index=rng)
ts.resample('H').agg(['count'])
print('ok')
