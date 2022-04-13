#date: 2022-04-13T16:48:32Z
#url: https://api.github.com/gists/fcdcb2e52d77e41ab8c8455067643185
#owner: https://api.github.com/users/salsferrazza

#!/bin/bash

# NOTE:
# - the host environment requires these CLIs to be installed: json, envsubst, httpie
# - this has only been tested on OS X
#
# contact: sal@instigate.net

export USER_ID= # don't have API access yet
export API_PREFIX="https://trading-api.kalshi.com/v1/cached"
export TODAY=$(date "+%y%b%d" | awk '{print toupper($0)}')
export PREFIX="HIGHNY"

# just pick the last market we see in the list
export MARKET_ID=$(http "${API_PREFIX}/events/${PREFIX}-${TODAY}" | json -a .event.markets| json -a .id |tail -1) 
export TICKER=$(http "${API_PREFIX}/markets/${MARKET_ID}" | json .market.ticker_name)

# there were no instructions on what to do with the book 
# once it was queried, but I've built order books from 
# real-time market data connections over FIX before
export BOOK=$(http "${API_PREFIX}/markets/${MARKET_ID}/order_book") 

# templatize the request body for new orders
read -r -d '' ORDER_TEMPLATE <<'EOF'
{"count": 1,
"market_id": "${MARKET_ID}",
"price": 25,
"side": "yes"}
EOF

# this is the request body that I would manufacture to 
# POST an order to the API based on the market ID
export REQUEST=$(echo ${ORDER_TEMPLATE} | envsubst)
echo $REQUEST

# pretend we're executing httpie here for the new order (with request body)
# http POST ...
# *must* ensure a 2xx class response and order_id in the response body
# might want to consider retry mechanism until a 2xx is received or the trader aborts

# I can't place a real order without API access
# but if I could, I'd process the order_id from the 
# response body once I get a 2xx class response
export ORDER_ID=$(echo ${RESPONSE} | json .order.order_id)
echo $ORDER_ID

sleep 1

# pretend we're executing httpie here for the cancellation
# http DELETE https://trading-api.kalshi.com/v1/users/${USER_ID}/orders/${ORDER_ID}

# NOTE: removing the order entails calling HTTP DELETE on the  
# user and order endpoint combination. I wouldn't consider this 
# order properly cancelled until I successfully received an
# HTTP 2xx class response. Would probably want to implement an 
# exponential backoff retry mechanism for cases where the API
# endpoint is unavailable or returning 5xx or 4xx multiple times
# consecutively. 

# NOTE: position management can be a complicated topic. I don't see a Kalshi
# drop copy API, but that would be great for out-of-band risk management.
# Otherwise, systems that are responsible for position management will need to 
# reconcile fills coming back from Kalshi with what Kalshi attests are a trader's 
# posiitons from its own accounting system. Some of the work I've done in this area 
# (P&L viz, real-time market data, event-driven order management) can be seen at:
# https://g.co/cloud/marketdata
