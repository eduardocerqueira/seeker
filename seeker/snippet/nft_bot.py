#date: 2022-09-05T17:16:55Z
#url: https://api.github.com/gists/7affb91ee5fd0bdb1ef578c7dd618372
#owner: https://api.github.com/users/puckzhengmath

import requests
from bs4 import BeautifulSoup
import pandas as pd
import hashlib

def getNft(count):
    resp = requests.get(f'https://www.coingecko.com/en/nft?page={count}')
    soup = BeautifulSoup(resp.text, 'html.parser')
    table = soup.find('table', {'class': 'sort table mb-0 text-sm text-lg-normal table-scrollable'})

    if table:
        rows = table.find_all('tr')
        colname = [x.get_text(strip=True) for x in rows[0].find_all('th')]
        href = [["https://www.coingecko.com"+y['href'] for y in x.find_all('a')] for x in rows[1:]]
        img = [[y['src'] for y in x.find_all('img')] for x in rows[1:]]
        data = [{col: row.get_text(strip=True) for col, row in zip(colname, x.find_all('td'))} for x in rows[1:]]

        for x,y,z in zip(data, img, href):
            x['img'] = y[0]
            x['href'] = z[0]
        return pd.DataFrame(data)
    else:
        return pd.DataFrame()

def hashing(key):

    return hashlib.md5(key.encode('utf-8')).hexdigest()

df = []
for i in range(1, 10):
    data = getNft(i)
    if not data.empty:
      df.append(data)
    else:
        break

df = pd.concat(df)
df['hash'] = df['NFT'].apply(lambda x : hashing(x))
print()