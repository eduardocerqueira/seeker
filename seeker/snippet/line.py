#date: 2024-09-26T17:07:24Z
#url: https://api.github.com/gists/c38101102965ac9fd743118324cf5fe1
#owner: https://api.github.com/users/LinuxIsCool

import pandas as pd
import holoviews as hv
import hvplot.pandas
hv.Points((1,1)).opts(size=10) * hv.Points((2,2)).opts(size=10) * pd.DataFrame([[1,1],[2,2]]).hvplot.line(x='0',y='1')