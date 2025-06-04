#date: 2025-06-04T16:51:20Z
#url: https://api.github.com/gists/35bbead431472da7d4439ab258b79b7b
#owner: https://api.github.com/users/Anish2

from bs4 import BeautifulSoup
import requests
import uuid
import hashlib

### Fetch book text from API
res = requests.get(f'https://jainqq.org/booktext/exmple/{SRNO}') # replace SRNO with srno of book
soup = BeautifulSoup(res.text, 'html.parser')
booktext = soup.get_text()
chunks = re.split(r'Page #\d+', booktext)
chunks.pop(0)
# chunks is all the page texts