#date: 2022-10-04T17:08:48Z
#url: https://api.github.com/gists/13c76ec1c472a2a071613ec62e1866ea
#owner: https://api.github.com/users/OnepagecodeSource

symbol="AAPL" #AAPL is the symbol for Apple Inc.

URL=('https://yfapi.net/ws/insights/v1/finance/insights?'
     'symbol={}'
    )
URL=URL.format(symbol)
header = {
    'X-API-KEY': "{{API_KEY}}"
    }
    
response = requests.request("GET", URL, headers=header).json()
print(json.dumps(response, indent=4))