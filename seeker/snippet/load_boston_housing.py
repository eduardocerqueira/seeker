#date: 2022-04-05T16:53:00Z
#url: https://api.github.com/gists/209917cc778373daef90ccd5623c7e44
#owner: https://api.github.com/users/SharpKoi

import numpy as np
import pandas as pd

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# the description of the columns is listed in `data_url`.
columns = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat', 'mediv']
df = pd.DataFrame(np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :3]]), 
                  columns=columns)

df.head()