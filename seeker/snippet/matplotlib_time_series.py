#date: 2021-09-24T17:13:18Z
#url: https://api.github.com/gists/cd72d21b0635b2607336cd998a88dbac
#owner: https://api.github.com/users/zjwarnes

import mplfinance as mpf
mpf.plot(df.sort_index(ascending=True),type='candle', title='Bitcoin Price')