#date: 2023-10-12T17:03:08Z
#url: https://api.github.com/gists/8970b8ce67bc84dc5047ab729922d44d
#owner: https://api.github.com/users/fengtality

import pandas as pd
import os

def main():
    """
    This script processes a Hummingbot trades CSV file containing trade data and generates metrics such as total volume, 
    buy/sell amount, average prices, and profitability for each symbol. The results are saved to a new CSV file and displayed.
    """
    
    # Prompt the user for the file path
    data_path = input("Please enter the path to the CSV file: ")

    # Load the data
    data = pd.read_csv(data_path)

    # Compute trade volume for each row
    data['trade_volume'] = data['price'] * data['amount']

    # Extract relevant data by trade type
    buy_data = data[data['trade_type'] == 'BUY']
    sell_data = data[data['trade_type'] == 'SELL']

    # Compute metrics for buys
    total_buy_volume_by_symbol = buy_data.groupby('symbol')['trade_volume'].sum()
    total_buy_amount_by_symbol = buy_data.groupby('symbol')['amount'].sum()

    # Compute metrics for sells
    total_sell_volume_by_symbol = sell_data.groupby('symbol')['trade_volume'].sum()
    total_sell_amount_by_symbol = sell_data.groupby('symbol')['amount'].sum()

    # Compute average prices
    weighted_avg_buy_price_by_symbol = (total_buy_volume_by_symbol / total_buy_amount_by_symbol).fillna(0)
    weighted_avg_sell_price_by_symbol = (total_sell_volume_by_symbol / total_sell_amount_by_symbol).fillna(0)

    # Additional metrics
    fees_paid_by_symbol = data.groupby('symbol')['trade_fee_in_quote'].sum()
    last_price_by_symbol = data.groupby('symbol').last()['price']
    difference_amount_by_symbol = total_buy_amount_by_symbol - total_sell_amount_by_symbol
    value_of_unsold_assets_by_symbol = difference_amount_by_symbol * last_price_by_symbol
    total_matched_volume = total_buy_volume_by_symbol + total_sell_volume_by_symbol - abs(value_of_unsold_assets_by_symbol)
    profitability_by_symbol = total_sell_volume_by_symbol - total_buy_volume_by_symbol + value_of_unsold_assets_by_symbol

    # Construct summary dataframe
    summary_df_symbol = pd.DataFrame({
        'Trades': data.groupby('symbol').size(),
        'Total Volume': data.groupby('symbol')['trade_volume'].sum(),
        'Buy Amount': total_buy_amount_by_symbol,
        'Sell Amount': total_sell_amount_by_symbol,
        'Avg Buy Price': weighted_avg_buy_price_by_symbol,
        'Avg Sell Price': weighted_avg_sell_price_by_symbol,
        'Last Price': last_price_by_symbol,
        'Buy Volume': total_buy_volume_by_symbol,
        'Sell Volume': total_sell_volume_by_symbol,
        'Fees Paid': fees_paid_by_symbol,
        'Unsold Assets': value_of_unsold_assets_by_symbol,
        'Matched Volume': total_matched_volume,
        'Profit': profitability_by_symbol
    })

    # If only one symbol, update price fields for total
    if len(data['symbol'].unique()) == 1:
        single_symbol = data['symbol'].unique()[0]
        avg_buy_price_total = summary_df_symbol.at[single_symbol, 'Avg Buy Price']
        avg_sell_price_total = summary_df_symbol.at[single_symbol, 'Avg Sell Price']
        last_price_total = summary_df_symbol.at[single_symbol, 'Last Price']
    else:
        avg_buy_price_total = avg_sell_price_total = last_price_total = None

    # Add totals to the dataframe
    summary_df_symbol.loc['Total'] = summary_df_symbol.sum(numeric_only=True)
    summary_df_symbol.at['Total', 'Avg Buy Price'] = avg_buy_price_total
    summary_df_symbol.at['Total', 'Avg Sell Price'] = avg_sell_price_total
    summary_df_symbol.at['Total', 'Last Price'] = last_price_total

    # Round specific columns to two decimals
    cols_to_round = ['Total Volume', 'Buy Amount', 'Sell Amount', 'Buy Volume', 'Sell Volume', 'Fees Paid', 'Unsold Assets', 'Matched Volume', 'Profit']
    summary_df_symbol[cols_to_round] = summary_df_symbol[cols_to_round].round(2)

    # Extract filename from the provided path
    filename = os.path.basename(data_path)

    # Check for and create 'metrics' directory if it doesn't exist
    if not os.path.exists('metrics'):
        os.makedirs('metrics')

    # Construct the output path using the 'metrics' directory
    output_path = os.path.join('metrics', filename.replace('.csv', '-metrics.csv'))

    # Save the dataframe to the new CSV path
    summary_df_symbol.to_csv(output_path)

    # Print summary statistics
    print("\nSummary by Symbol:\n", summary_df_symbol)
    print(f"\nMetrics saved to: {output_path}")

if __name__ == "__main__":
    main()
