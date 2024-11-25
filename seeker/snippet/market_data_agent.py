#date: 2024-11-25T16:57:39Z
#url: https://api.github.com/gists/0dcfe46dcaec5a5bcc5e5dc292fd8eb7
#owner: https://api.github.com/users/amrlxamn

import requests
import pandas as pd
import time

# Replace 'YOUR_API_KEY' with your TraderMade API key
API_KEY = 'YOUR_API_KEY'
CURRENCY_PAIRS = 'EURUSD,GBPUSD'  # Modify as needed
API_URL = 'https://marketdata.tradermade.com/api/v1/live'

def fetch_live_data():
    """Fetch live Forex rates from TraderMade API."""
    params = {
        'currency': CURRENCY_PAIRS,
        'api_key': API_KEY
    }
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        return data['quotes']
    except requests.exceptions.HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')
    return None

def process_data(quotes):
    """Process and display the fetched data."""
    df = pd.DataFrame(quotes)
    print(df)

def main():
    """Main function to run the Market Data Agent."""
    while True:
        quotes = fetch_live_data()
        if quotes:
            process_data(quotes)
        else:
            print('Failed to retrieve data.')
        time.sleep(60)

if __name__ == '__main__':
    main()
