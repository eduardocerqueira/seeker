#date: 2022-12-29T16:28:28Z
#url: https://api.github.com/gists/4d7e8fc8dcfaad1dee3ea671ef8c680a
#owner: https://api.github.com/users/matrixise

from bs4 import BeautifulSoup

soup = BeautifulSoup(open("index.html"))
soup = BeautifulSoup("<html>data</html>")
