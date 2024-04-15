#date: 2024-04-15T16:55:05Z
#url: https://api.github.com/gists/d06c26853f854fd2e6ab29a248038eca
#owner: https://api.github.com/users/ksolana

import requests
import pandas as pd
from datetime import datetime
from dateutil import parser
import time

def datetime_parse(dt_string):
    #dt_string example: "December 30, 2023 13:06:57 UTC"
    dt = parser.parse(dt_string).timestamp()
    return dt

def datetime_to_timestamp():

    dt_string = "12/11/2018 09:15:32"
    dt_string = "December 30, 2023 13:06:57"

    # Considering date is in mm/dd/yyyy format
    dt_object2 = datetime.strptime(dt_string, "%B %d, %Y %H:%M:%S")
    print("dt_object2 =", dt_object2.timestamp())

def print_df_ohlc(ohlc):
    df = pd.DataFrame(ohlc)
    df.columns = ['date', 'open', 'high', 'low', 'close']
    df['date'] = pd.to_datetime(df['date'], unit = 'ms')
    df.set_index('date', inplace = True)
    print(df)

def print_closest_price(ohlc, ts):
    tms = ts*1000
    df = pd.DataFrame(ohlc)
    df.columns = ['prices', 'market_caps', 'total_volumes']
    closest = 0
    diff = tms
    for p in df['prices']:
        dx = abs(tms - p[0])
        if dx < diff:
            closest = p
    return closest

def print_df_prices_mean(ohlc):
    df = pd.DataFrame(ohlc)
    df.columns = ['prices', 'market_caps', 'total_volumes']
    def f(x):
        # Get the price element only otherwise it looks like '[1703934012803, 102.07297851054362]'
        return x[1]
    df['prices'] = df['prices'].apply(f)
    #print('prices size:', len(df['prices']))
    # Mean Max and Min
    return [df['prices'].mean(), df['prices'].max(), df['prices'].min()]

def get_dates_of_stake_prices():
    dates = ["December 30, 2023 13:06:57 UTC",
    "December 28, 2023 06:24:48 UTC",
    "December 26, 2023 00:20:52 UTC",
    "December 23, 2023 18:19:16 UTC",
    "December 21, 2023 12:11:08 UTC",
    "December 19, 2023 05:31:12 UTC",
    "December 16, 2023 22:30:55 UTC",
    "December 14, 2023 15:53:48 UTC",
    "December 12, 2023 08:28:16 UTC",
    "December 10, 2023 01:17:21 UTC",
    "December 7, 2023 18:40:20 UTC",
    "December 5, 2023 12:49:54 UTC",
    "December 3, 2023 08:23:29 UTC",
    "December 1, 2023 04:31:54 UTC",
    "November 28, 2023 23:53:02 UTC",
    "November 26, 2023 19:18:24 UTC",
    "November 24, 2023 15:47:39 UTC",
    "November 22, 2023 11:28:13 UTC",
    "November 20, 2023 07:15:08 UTC",
    "November 18, 2023 04:42:13 UTC",
    "November 16, 2023 00:13:35 UTC",
    "November 13, 2023 19:38:35 UTC",
    "November 11, 2023 15:07:17 UTC"]
    return dates

def get_coin_market_chart_range_by_id(ts):
    from pycoingecko import CoinGeckoAPI
    cg = CoinGeckoAPI()
    # Get the prices of a couple of hours around that time.
    # We can then either compute average, or get the nearest timestamp.
    ohlc = cg.get_coin_market_chart_range_by_id(id = 'solana', vs_currency='usd', from_timestamp=ts-7000,
                                                to_timestamp=ts+7000)
    #return print_df_prices_mean(ohlc)
    return print_closest_price(ohlc, ts)

def get_all_stake_prices():
    dates = get_dates_of_stake_prices()
    for d in dates:
        ts = int(datetime_parse(d))
        time.sleep(10)
        p = get_coin_market_chart_range_by_id(ts)
        print(f"{d},{ts},{p}")
        #break

get_all_stake_prices()


'''
Test code for CoinGecko
'''
def get_cg_price():
    from pycoingecko import CoinGeckoAPI
    cg = CoinGeckoAPI()
    ohlc = cg.get_coin_ohlc_by_id(id = 'solana', vs_currency = 'usd', days = '1')
    #print (ohlc)

def get_inst_price():
    url = 'https://api.coingecko.com/api/v3/simple/price'
    params = {  
             'ids': 'solana',
             'vs_currencies': 'USD'
    }

    response = requests.get(url, params = params)
    if response.status_code == 200:
             data = response.json()
             Ethereum_price = data['ethereum']['usd']
             print(f'The price of Ethereum in USD is ${Ethereum_price}')
    else:
             print('Failed to retrieve data from the API')
