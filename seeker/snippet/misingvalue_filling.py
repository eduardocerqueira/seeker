#date: 2023-09-21T17:08:41Z
#url: https://api.github.com/gists/43c93d533630e728c9f62912d1f6648c
#owner: https://api.github.com/users/maacz

import numpy as np
import pandas as pd
ab=pd.read_csv('/home/user/Downloads/customer1.txt',header=None)
ab.columns=['id','fname','lname','age','prof','location']
print(ab)
ab1=ab.fillna('india')