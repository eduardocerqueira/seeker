#date: 2025-08-08T17:08:04Z
#url: https://api.github.com/gists/c106af96c19b0d791e447e39bf52068d
#owner: https://api.github.com/users/datavudeja

"""
Defining a custom function to be applied in pandas groupby
"""

import numpy as np
import pandas as pd

clients = ['joao', 'joao', 'joao', 'lucas', 'lucas', 'julia', 'julia', 'julia', 'julia']
products = ['smartphone', 'notebook', 'book', 'ball', 'car', 'hat', 'bike', 'mouse', 'pen']

df = pd.DataFrame({'clients': clients, 
                   'products': products})

""" print(df):
------------------------
|clients | products    |
------------------------
| joao    | smartphone |
| joao    | notebook   |
| joao    | book       |
| lucas   | ball       |
| lucas   | car        |
| julia   | hat        |
| julia   | bike       |
| julia   | mouse      |
| julia   | pen        |
---------------------
"""

# applying a custom function to get the list of products for client
def get_products(series):
    list_of_products = ','.join(series)
    return list_of_products
  
df['list_of_products'] = df.groupby('clients').products.transform(get_products)

""" print(df):
----------------------------------------------------
|clients | products    | list_of_products          |
----------------------------------------------------
| joao    | smartphone | smartphone,notebook, book |
| joao    | notebook   | smartphone,notebook, book |
| joao    | book       | smartphone,notebook, book |
| lucas   | ball       | ball,car                  |
| lucas   | car        | ball,car                  |
| julia   | hat        | hat,bike,mouse,pen        |
| julia   | bike       | hat,bike,mouse,pen        |
| julia   | mouse      | hat,bike,mouse,pen        |
| julia   | pen        | hat,bike,mouse,pen        |
----------------------------------------------------
"""
