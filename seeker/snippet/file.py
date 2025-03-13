#date: 2025-03-13T16:51:15Z
#url: https://api.github.com/gists/e9fabc1474dcca9bd4ce74d8d0172947
#owner: https://api.github.com/users/NHasan143

import requests
from bs4 import BeautifulSoup

url = 'https://example.com'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')