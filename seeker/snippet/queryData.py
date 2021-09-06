#date: 2021-09-06T17:08:37Z
#url: https://api.github.com/gists/c5d739500cc07aa643d9ce4549281413
#owner: https://api.github.com/users/martinkarlssonio

"""
Written by Martin Karlsson
www.martinkarlsson.io

How to perform a SQL Query on your data in Python.
"""
# Import pandas and pandassql
import pandas as pd
from pandasql import sqldf

# Some lists with data
epochList = [1630946340,1630945340,1630944340,1630943340]
priceList = [67.9, 23.5, 98.6, 754.7]
customerIdList = ["customerQ", "customerW", "customerE", "customerR"]

# Place your data in a dict
aDict = {'epoch': epochList, 'price': priceList, 'customerId': customerIdList}

# Create a Pandas dataframe from your dict
aDataframe = pd.DataFrame(aDict, columns=aDict.keys())

# Utilize 'pandasql' and perform a query to fetch the data of interest
customerQTransactions = sqldf("SELECT * FROM aDataframe WHERE customerId = 'customerQ' AND price > 2")
print(customerQTransactions)

"""
OUTPUT:
        epoch  price customerId
0  1630946340   67.9  customerQ
"""
