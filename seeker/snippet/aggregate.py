#date: 2025-08-08T16:51:04Z
#url: https://api.github.com/gists/e6f944d4f1f60c4ac73d32d0b8ddf7f0
#owner: https://api.github.com/users/datavudeja

# How to apply list of functions on pd.Dataframe# How t 
# https://stackoverflow.com/questions/48767067/how-to-pass-list-of-custom-functions-to-pandas-dataframe-aggregate

import pandas as pd
import numpy as np
from scipy.stats import trim_mean

df = pd.DataFrame(np.random.randn(100, 3), columns=['A', 'B', 'C'], index=pd.date_range('1/1/2000', periods=100))

# this works as expected
print(df.agg([np.sum, np.mean]))

# now with a different function, works also
print(df.agg(lambda x: trim_mean(x, 0.2)))

# apply also works
print(df.apply(lambda x: trim_mean(x, 0.2)))

# now with list of lambda: doesn't work
# df.apply([lambda x: trim_mean(x, 0.2)]) # tuple index out of range
# df.agg([lambda x: trim_mean(x, 0.2)]) # tuple index out of range


# solution: wrap list of functions in lambda# solut 
c = ['trim_mean','mean','sum']
print (df.agg(lambda x: pd.Series([trim_mean(x, 0.2), np.mean(x), np.sum(x)], index=c)))