#date: 2023-07-17T16:44:11Z
#url: https://api.github.com/gists/e45a1fbdbd9a6536ed36c199b1741af8
#owner: https://api.github.com/users/mwestwood

import pandas as pd

# Sample data
data = {
    'error_budget_remaining_percent': [90, 85, 80, 75],
    'ebbr': [0.05, 0.02, 0.1, 0.08],
    'error_count': [10, 5, 15, 8],
    'timestamp': pd.to_datetime(['2023-07-01', '2023-07-02', '2023-07-03', '2023-07-04'])
}

df = pd.DataFrame(data)

def total_errors_in_last_n_days(dataframe, n):
    # Get the maximum timestamp in the dataframe
    max_timestamp = dataframe['timestamp'].max()

    # Calculate the start date for the last n days
    start_date = max_timestamp - pd.Timedelta(days=n)

    # Filter the dataframe for the last n days
    last_n_days_data = dataframe[dataframe['timestamp'] >= start_date]

    # Calculate the total number of errors in the last n days
    total_errors = last_n_days_data['error_count'].sum()

    return total_errors

# Usage
n_days = 3
total_errors_last_n_days = total_errors_in_last_n_days(df, n_days)
print("Total errors in the last {} days: {}".format(n_days, total_errors_last_n_days))
