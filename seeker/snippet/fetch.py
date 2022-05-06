#date: 2022-05-06T17:20:29Z
#url: https://api.github.com/gists/78d271263332b51ff4798e2de6c8baf0
#owner: https://api.github.com/users/meooow25

import re
import requests
from Crypto.Cipher import AES

user_agent = 'Mozilla/5.0 AppleWebKit/537.36 Chrome/102.0.4972.0 Safari/537.36'
home_page = 'https://codeforces.com'

def init_session(s):
    s.headers.update({'User-Agent': user_agent})
    r = s.get(home_page)
    r.raise_for_status()
    html = r.text
    if '/aes.min.js' in html:
        rcpc = get_rcpc(html)
        s.cookies.set('RCPC', rcpc, domain='codeforces.com', path='/')
        r = s.get(home_page)
        r.raise_for_status()
        html = r.text
    csrf = re.search('name="X-Csrf-Token" content="([0-9a-f]+)"', html).group(1)
    s.headers.update({'X-Csrf-Token': csrf})

def get_rcpc(html):
    def getvar(var):
        return bytes.fromhex(re.search(var + '=toNumbers\("([0-9a-f]+)"\)', html).group(1))
    key, iv, ciphertext = getvar('a'), getvar('b'), getvar('c')
    return AES.new(key, AES.MODE_CBC, iv=iv).decrypt(ciphertext).hex()

s = requests.Session()
init_session(s)

r = s.post('https://codeforces.com/data/submitSource', data={'submissionId': '154770308'})
print(r.json()['source'])
