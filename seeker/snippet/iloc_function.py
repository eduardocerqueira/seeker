#date: 2023-09-21T17:01:14Z
#url: https://api.github.com/gists/e9c6c91dd60d37a8a904871aedc1522a
#owner: https://api.github.com/users/maacz

import numpy as np
import pandas as pd
df=pd.read_csv('/home/user/Downloads/customer1.txt',header=None)
df.columns=['id1','fname','lname','age','prof','location']
df1=df.iloc[1:2,1:3]
print(df1)
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
print(y)