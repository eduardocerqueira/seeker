#date: 2022-03-28T17:01:45Z
#url: https://api.github.com/gists/159b569d1dcb36a138d4e55ce3049682
#owner: https://api.github.com/users/brulint


import time
import hmac, hashlib
import requests

def sign():
    user = '' # username @ Bitstamp
    key = '' # API key
    secret = '' # API secret
    now = time.strftime("%s")
    message = now + user + key
    signature = hmac.new(secret.encode(),message.encode(),hashlib.sha256).hexdigest().upper()
    return {'key':key,'signature':signature,'nonce':now}

uri = "https://www.bitstamp.net/api/v2/"

# public requests

requests.get(uri + 'ticker/btceur').json()
requests.get(uri + 'ohlc/btceur', {'limit': 1000, 'step': 3600}).json()

# private requests

requests.post(uri + 'balance/btceur/', sign()).json()
requests.post(uri + 'buy/instant/btceur/', {**sign(), 'amount': 10}).json() # amount in EUR
requests.post(uri + 'cancel_order/', {**sign(), 'id': order_id}).json()
