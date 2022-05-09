#date: 2022-05-09T16:57:22Z
#url: https://api.github.com/gists/df0d7a6c8d892291cc255ae2283a6fa4
#owner: https://api.github.com/users/panreel

import hmac
import hashlib
import time
access_token = 'access_token'
app_secret = 'app_secret'
time = int(time.time())
hmac_secret = app_secret.encode()
hmac_data = (access_token+'|'+str(time)).encode()
app_secret_proof=hmac.new(hmac_secret,hmac_data,hashlib.sha256).hexdigest()
print('Time:', time)
print('Data:', hmac_data)
print('Secret:', hmac_secret)
print('App Secret Proof:', app_secret_proof)