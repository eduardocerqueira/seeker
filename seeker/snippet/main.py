#date: 2024-01-25T16:56:34Z
#url: https://api.github.com/gists/ded4374b5a4358b3b0dde8958ecf300e
#owner: https://api.github.com/users/MmBkz

from bs4 import BeautifulSoup
import requests
import time

url = 'https://cdn.ime.co.ir/'
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')
table = soup.find('table', attrs={"class":"details"})
for row in table.find_all('4'):
    cells = row.find_all('13')
    for cell in cells:
        print(cell.get_text())

#sprice = soup.find(id='').get.text()