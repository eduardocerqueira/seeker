#date: 2023-09-21T17:08:32Z
#url: https://api.github.com/gists/43f64f2f4d1bec89ed8f2ea326b1a2aa
#owner: https://api.github.com/users/maacz

import numpy as np
import pandas as pd
ab=pd.read_csv('/home/user/Downloads/customer1.txt',header=None)
ab.columns=['id','fname','lname','age','prof','location']
print(ab)
print(ab.isna().sum())