#date: 2022-03-14T17:08:47Z
#url: https://api.github.com/gists/58b684e1405c337b8ad926b9d642614b
#owner: https://api.github.com/users/SebastianUrdaneguiBisalaya

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as wb

import datetime
start = datetime.datetime(2021,1,1)
end = datetime.datetime(2022,2,28)
def getStocks(stock):
    df = pd.DataFrame()
    for element in stock:
        df[element] = wb.DataReader(element, data_source='yahoo', 
                            start=start, end=end)['Adj Close']
    return df

list_stocks = ['TSLA', 'ABNB', 'AAPL', 'MSFT', 'GOOG', 'FB', 
               'AMZN', 'BABA', 'VWAGY', 'AEO', 'JPM', 'PYPL', 
               'NVDA']
stocks = getStocks(list_stocks)
stocks