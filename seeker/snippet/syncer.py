#date: 2022-05-30T17:09:38Z
#url: https://api.github.com/gists/5c754368dea7c18dadecb33eede73df1
#owner: https://api.github.com/users/mehrfirouzian

import requests, os

from bs4 import BeautifulSoup


extensions = ['flac', 'mp3', 'lrc', 'dff']  # sample extensions for music

os.chdir('/home/mehrshad/Music')
files = os.listdir()

address = input('Enter address: ')
if not address.endswith('/'):
    address += '/'

if not address.startswith('http'):
    address = 'http://' + address
def fetch(url):
    soup = BeautifulSoup(requests.get(url).content, features="html.parser")
    for a in soup.find_all('a'):
        if a.text[a.text.rfind('.')+1:].lower() in extensions:
            if a.text not in files:
                yield url + a.attrs['href']
        elif a.text.endswith('/'):
            for f in fetch(url+a.attrs['href']):
                yield f

for song in fetch(address):
    os.system(f'wget "{song}"')
