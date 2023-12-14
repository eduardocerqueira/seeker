#date: 2023-12-14T16:48:37Z
#url: https://api.github.com/gists/af37f14d6ef70a365b995cc64110d40e
#owner: https://api.github.com/users/jimtin

import alpaca_interactions as alpaca
import datetime
import indicators

# List of symbols
symbols = ["AAPL", "GOOGL", "META"]
max_number_of_candles = 1000
timeframe = "1hour"

# Function to run the trading bot
def auto_run_trading_bot():
    """
    Function to run the trading bot
    """
    # Print Welcome to your very own trading bot
    print("Welcome to your very own trading bot")
    # Set the end date to yesterday
    end_date = datetime.datetime.now() - datetime.timedelta(days=1) # Note that if you have a premium subscription you can remove this restriction
    # Set the start date to one year ago
    start_date = end_date - datetime.timedelta(days=365)
    #### Calculate the RSI ####
    for symbol in symbols:
        # Convert symbol to a list
        symbol = [symbol]
        # Get the historical data for the symbol
        symbol_historical_data = alpaca.get_historic_bars(
            symbols=symbol, 
            timeframe=timeframe, 
            start_date=start_date, 
            end_date=end_date, 
            limit=max_number_of_candles
        )
        # Calculate the RSI for the symbol
        rsi_data = indicators.calc_indicator(
            indicator_name="rsi",
            historical_data=symbol_historical_data,
            rsi_period=14,
            rsi_high=70,
            rsi_low=30
        )
        # Print the RSI data
        print(rsi_data)


# Main function for program
if __name__ == "__main__":
    auto_run_trading_bot()