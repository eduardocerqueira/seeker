#date: 2021-09-24T17:13:18Z
#url: https://api.github.com/gists/cd72d21b0635b2607336cd998a88dbac
#owner: https://api.github.com/users/zjwarnes

import requests
API_KEY = 'your API key here'
url = 'https://www.alphavantage.co/query?function=CRYPTO_INTRADAY&symbol=BTC&market=USD&interval=1min&outputsize=full&apikey='+API_KEY
r = requests.get(url)
data = r.json()

df = pd.DataFrame.from_dict(data['Time Series Crypto (1min)']).T
df = df.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. volume': 'volume'})
df.index = pd.to_datetime(df.index)
df =df.astype(float)