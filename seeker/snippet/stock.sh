#date: 2022-02-03T17:11:26Z
#url: https://api.github.com/gists/af96c72c4fc81f897194af09c40552a7
#owner: https://api.github.com/users/areese801

#!/bin/bash

###
### Gets current price of a stock.
### See:  https://hackernoon.com/you-can-track-stock-market-data-from-your-terminal-1k1h3135
###


tickerSymbol=${1}

if [ -z "${tickerSymbol}" ]
then
	echo "You must supply a ticker symbol."
	return
fi

tickerSymbol=$(echo "${tickerSymbol}" | tr '[:lower:]' '[:upper:]')

urlBase=https://terminal-stocks.herokuapp.com
url=${urlBase}/${tickerSymbol}

curl ${url}