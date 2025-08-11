#date: 2025-08-11T17:13:41Z
#url: https://api.github.com/gists/766b8e16436705f6b7368d846cd0b845
#owner: https://api.github.com/users/blueman-hacker

import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import numpy as np
import matplotlib.dates as mdates
import io
import os

output = io.StringIO()  # Create a string buffer to store output for PDF generation

def print_f(*args, **kwargs):
    new_args = []
    for arg in args:
        if isinstance(arg, float):
            formatted = f"{arg:,.2f}"
        elif isinstance(arg, int):
            formatted = f"{arg:,}"
        else:
            formatted = str(arg)
        new_args.append(formatted)
    line = " ".join(new_args)
    print(line, **kwargs)

stocks = {}
print_f("Please insert your stocks, the price you bought them for, and the date you bought them (YYYY-MM-DD).")
print_f("Type 'done' when finished.\n")

while True:
    symbol = input("Enter stock symbol (or type 'done' to finish): ").strip().upper()
    if symbol.lower() == 'done':
        break

    try:
        price = float(input(f"Enter the price you bought {symbol} at: "))
        date = input(f"Enter the date you bought {symbol} (YYYY-MM-DD): ").strip()
        datetime.strptime(date, "%Y-%m-%d")  # Validate date
        shares = float(input(f"Enter how many shares of {symbol} you bought: "))
        stocks[symbol] = {"bought_price": price, "bought_date": date, "shares": shares}

    except ValueError:
        print("Invalid input. Please try again.\n")

def get_percentage_gain(current, past):
    return ((current - past) / past) * 100

print_f("\n===============================================================================")
print_f("Your portfolio is formed by the following:", list(stocks.keys()) if stocks else 'No stocks in portfolio', "\n")

total_portfolio_value = 0.0

for symbol, info in stocks.items():
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="300mo")

    current_price = hist['Close'][-1]
    total_portfolio_value += current_price * info['shares']
    today = datetime.today().date()
    this_week = today - timedelta(days=today.weekday())
    week_ago = today - timedelta(days=7)
    one_month_ago = today - timedelta(days=30)
    bought_date = datetime.strptime(info['bought_date'], "%Y-%m-%d").date()
    bought_price = info['bought_price']

    price_today = current_price
    price_week_start = hist.loc[str(this_week):]['Open'][0]
    price_week_ago = hist.loc[str(week_ago):]['Open'][0]
    price_month_ago = hist.loc[str(one_month_ago):]['Close'][0]

    print_f(f"\n================================== {symbol} ==================================")
    formatted_bought_date = bought_date.strftime("%B %d, %Y")
    print_f("Bought Price: $", bought_price, "on", formatted_bought_date, "or (", info['bought_date'], ") with the following date format: YYYY-MM-DD")
    print_f("Number of Shares:", info['shares'])
    print_f("Total Invested: $", bought_price * info['shares'])
    print_f("Current Price: $", price_today)
    print_f("Total Current Value: $", current_price * info['shares'])
    print_f("===============================================================================")

    if len(hist) >= 2:
        yesterday_close = hist['Close'][-2]
        print_f("Yesterday's closing price: $", yesterday_close)
    else:
        print_f("Not enough data to get yesterday's close.")

    print_f("% Gain/Loss Since Start of Day:", get_percentage_gain(price_today, yesterday_close), "%")
    print_f("% Gain/Loss Since Start of Week:", get_percentage_gain(price_today, price_week_start), "%")
    print_f("% Gain/Loss Over 1 Month:", get_percentage_gain(price_today, price_month_ago), "%")
    print_f("% Gain/Loss Over 1 Week:", get_percentage_gain(price_today, price_week_ago), "%")
    print_f("% Gain/Loss Over 1 Year:", get_percentage_gain(price_today, hist['Close'].iloc[-252]), "%")
    print_f("% Gain/Loss Since Bought:", get_percentage_gain(price_today, bought_price), "%")

    data = yf.download(symbol, period="6mo", interval="1d", progress=False)
    data['Close'].plot(title=f"{symbol} Stock Price Over Time", figsize=(10, 6), ylabel='Price (USD)')
    plt.show()


print_f("\n===============================================================================")
print_f("Total Portfolio Value: $", total_portfolio_value)
total_invested = sum(info['bought_price'] * info['shares'] for info in stocks.values())
print_f("Total Invested: $", total_invested)
print_f("Total Gain/Loss: $", total_portfolio_value - total_invested)
print_f("% Gain/Loss Since Bought:", get_percentage_gain(total_portfolio_value, total_invested), "%")
print_f("===============================================================================")

oldest_date = min(datetime.strptime(info['bought_date'], "%Y-%m-%d") for info in stocks.values())
today = datetime.today().date()
date_range = pd.date_range(start=oldest_date, end=today, freq='D')
portfolio_history = pd.DataFrame(index=date_range)

for symbol, info in stocks.items():
    purchase_date = datetime.strptime(info['bought_date'], "%Y-%m-%d")
    data = yf.download(symbol, start=purchase_date - timedelta(days=7), end=today + timedelta(days=1), interval="1d", progress=False)

    if data.empty:
        print_f("Skipping", symbol, "(no data).")
        continue

    data = data[['Close']].copy()
    data.rename(columns={'Close': symbol}, inplace=True)

    after_purchase = data.loc[purchase_date:]
    if after_purchase.empty or after_purchase[symbol].dropna().empty:
        print_f("Skipping", symbol, "(no price after purchase date).")
        continue

    data[symbol] = data[symbol] * info['shares']
    data = data.reindex(date_range)
    data[symbol] = data[symbol].ffill()
    portfolio_history[symbol] = data[symbol]

portfolio_history.dropna(how='all', inplace=True)
portfolio_history['Total Value'] = portfolio_history.sum(axis=1)
initial_value = portfolio_history['Total Value'].iloc[0]
portfolio_history['Cumulative Return (%)'] = (portfolio_history['Total Value'] / initial_value - 1) * 100

plt.figure(figsize=(10, 5))
plt.plot(portfolio_history.index, portfolio_history['Cumulative Return (%)'], label='Cumulative Return', color='green')
plt.title("Portfolio Cumulative Return Over Time")
plt.xlabel("Date")
plt.ylabel("Cumulative Return (%)")
plt.grid(True)
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()