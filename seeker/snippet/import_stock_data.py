#date: 2021-09-28T17:02:14Z
#url: https://api.github.com/gists/85cd3038f1992537bc57f64ee9af13a5
#owner: https://api.github.com/users/woutervanheeswijk

def read_stock_data(stock_symbol, start_date, end_date, interval):
    """Import price data from Yahoo Finance"""
    stock_data = web.get_data_yahoo(stock_symbol, start_date, end_date, interval=interval)

    return stock_data

stock_symbol = 'BKNG' # Stock symbol

# Set time period
start_date = datetime.datetime(2020, 1, 1)
end_date = datetime.datetime(2021, 9, 28)
delta = 'd' # Date interval, by default daily ('d')

stock_data = read_stock_data(stock_symbol, start_date, end_date, interval=delta)