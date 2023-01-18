#date: 2023-01-18T16:52:10Z
#url: https://api.github.com/gists/4a8f6ac49b9cd36dd321c49361c048b4
#owner: https://api.github.com/users/RiversideRocks

import requests
from bs4 import BeautifulSoup
from flask import Flask

def find():
    r = requests.get("https://my.frantech.ca/cart.php?gid=39").text

    soup = BeautifulSoup(r, 'html.parser')

    if int(soup.find_all("div", {"class": "package-qty"})[0].text.strip()[0]) > 0:
        return True
    else:
        return False

app = Flask(__name__)


@app.route('/')
def welcome():
    if find() == True:
        return "Available!"
    else:
        return "Nothing"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)