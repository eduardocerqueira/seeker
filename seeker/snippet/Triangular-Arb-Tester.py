#date: 2023-01-10T17:05:30Z
#url: https://api.github.com/gists/e26dffac8fd3e199e6312f99c8d3b7c3
#owner: https://api.github.com/users/Sotacan

import ccxt

# Define the exchanges and symbols you want to use in the arbitrage
exchange_1 = ccxt.binance({
    'rateLimit': 2000,
    'enableRateLimit': True,
    'apiKey': 'YOUR_API_KEY',
    'secret': "**********"
})
exchange_2 = ccxt.bittrex({
    'rateLimit': 2000,
    'enableRateLimit': True,
    'apiKey': 'YOUR_API_KEY',
    'secret': "**********"
})
exchange_3 = ccxt.poloniex({
    'rateLimit': 2000,
    'enableRateLimit': True,
    'apiKey': 'YOUR_API_KEY',
    'secret': "**********"
})

symbol = 'BTC/USDT'

# minimum take profit amount
min_profit = 0.005

# Load the order book for each exchange
order_book_1 = exchange_1.fetch_order_book(symbol)
order_book_2 = exchange_2.fetch_order_book(symbol)
order_book_3 = exchange_3.fetch_order_book(symbol)

# Find the best bid and ask prices for each exchange
bid_1 = order_book_1['bids'][0][0] if len(order_book_1['bids']) > 0 else None
ask_1 = order_book_1['asks'][0][0] if len(order_book_1['asks']) > 0 else None
bid_2 = order_book_2['bids'][0][0] if len(order_book_2['bids']) > 0 else None
ask_2 = order_book_2['asks'][0][0] if len(order_book_2['asks']) > 0 else None
bid_3 = order_book_3['bids'][0][0] if len(order_book_3['bids']) > 0 else None
ask_3 = order_book_3['asks'][0][0] if len(order_book_3['asks']) > 0 else None

# Get the trading fee and slippage percentage for each exchange
fee_1 = exchange_1.load_markets()[symbol]['maker']  
slippage_1 = 0.01
fee_2 = exchange_2.load_markets()[symbol]['maker']
slippage_2 = 0.02
fee_3 = exchange_3.load_markets()[symbol]['maker']
slippage_3 = 0.01

# Check for arbitrage opportunity by comparing bid and ask prices
if ask_1 and bid_2 and ask_2 and bid_3:
    # Calculate the potential profit from the arbitrage
    profit = bid_1 - ask_3
    # apply slippage to the prices
    bid_1 = bid_1*(1-slippage_1)
    ask_1 = ask_1*(1+slippage_1)
    bid_2 = bid_2*(1-slippage_2)
    ask_2 = ask_2*(1+slippage_2)
    bid_3 = bid_3*(1-slippage_3)
    ask_3 = ask_3*(1+slippage_3)

    # If there is a positive profit, execute the trade
    if profit > min_profit:
        # Calculate the amount of BTC to buy
        amount = min(order_book_1['asks'][0][1], order_book_2['bids'][0][1], order_book_3['asks'][0][1])
        # Print the end results of the trade, but don't actually place the orders
        profit = profit*(1 - fee_1 - fee_2 - fee_3)
        print(f"Simulated trade: Bought {amount} {symbol} at {ask_1} on {exchange_1.name} and Sold at {bid_2} on {exchange_2.name} and Bought {amount} {symbol} at {ask_3} on {exchange_3.name} for a profit of {profit} after fees, slippage and Minimum take profit {min_profit}")
    else:
        print(f"Profit {profit} is less than the Minimum take profit {min_profit}")
else:
    print("No arbitrage opportunity found.").")