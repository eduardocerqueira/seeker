#date: 2022-06-09T16:41:12Z
#url: https://api.github.com/gists/a86348f56c547db92ed42d07b7b3bb2d
#owner: https://api.github.com/users/eshaben

import pandas as pd

data = pd.read_csv('~/Downloads/export-address-token-0xd3dfd3ede74e0dcebc1aa685e151332857efce2d.csv')

receive = data.loc[data['From'] == '0xd3dfd3ede74e0dcebc1aa685e151332857efce2d', 'Value']
send = data.loc[data['To'] == '0xd3dfd3ede74e0dcebc1aa685e151332857efce2d', 'Value']

send_sum = 0
receive_sum = 0

for s in send:
    if ',' in s:
        s = s.replace(',', '')
        send_sum += float(s)

for r in receive:
    if ',' in r:
        r = r.replace(',', '')
        receive_sum += float(r)

print(receive_sum, send_sum)
