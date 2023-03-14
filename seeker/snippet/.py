#date: 2023-03-14T17:16:47Z
#url: https://api.github.com/gists/084c8cf2085f74a29474190180e03ee2
#owner: https://api.github.com/users/adityasbg

import pandas as pd 
import numpy as np 

n_rows =1000
constant=0

phis= [0,0.9,0.5,1,1.2,-0.5,-1.2 ]

ar_data = pd.DataFrame()

for phi in phis:
    numbers= np.zeros(n_rows)
    for  i in range (1, n_rows) :
      numbers[i]= constant+ (phi* numbers[i-1]) + np.random.uniform()

    ar_data[f'phi= {phi}']=numbers

ar_data.head()