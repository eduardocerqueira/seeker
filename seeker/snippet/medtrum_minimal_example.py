#date: 2025-03-21T17:03:04Z
#url: https://api.github.com/gists/b492e5c0b32e421d6fc9ee6f86d5f2dd
#owner: https://api.github.com/users/bruderjakob12

import requests
from datetime import datetime, timezone, timedelta

username = 'email@example.com'
password = "**********"

header = {
    'DevInfo': 'Android 12;Xiamoi vayu;Android 12',
    'AppTag': 'v=1.2.70(112);n=eyfo;p=android',
    'User-Agent': 'okhttp/3.5.0'
}
url = 'https://easyview.medtrum.eu/mobile/ajax/login'
data = {
    'apptype': 'Follow',
    'user_name': username, 
    'password': "**********"
    'platform': 'google',
    'user_type': 'M',
}

r =requests.post(url, data=data, headers=header)
print(r.text)
header['Cookie'] = r.headers['Set-Cookie']
url = 'https://easyview.medtrum.eu/mobile/ajax/logindata'
r2 = requests.get(url, headers=header)
print(r2.text)

yesterday = datetime.now()-timedelta(days=1)
now = datetime.now(timezone.utc)
et = now.strftime("%Y-%m-%d %H:%M:%S").replace(' ', '%20')
st = yesterday.strftime("%Y-%m-%d %H:%M:%S").replace(' ', '%20')

url = 'https://easyview.medtrum.eu/mobile/ajax/download?flag=sg&st='+st+'&et='+et+'&user_name=' + r2.json()['monitorlist'][0]['username']
r3 = requests.get(url, headers=header)
print(r3.text)