#date: 2021-09-09T17:02:59Z
#url: https://api.github.com/gists/82dba67f7e1f730ca2f147a263e35508
#owner: https://api.github.com/users/nickfogle

import hmac
import hashlib
email_hash = hmac.new(
  API_KEY, # Your API Key (keep safe)
  CUSTOMER_ID, # Stripe Customer ID
  digestmod=hashlib.sha256
).hexdigest() # Send to front-end