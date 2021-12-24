#date: 2021-12-24T17:19:52Z
#url: https://api.github.com/gists/e93ac7c14e10f7ae6952f6c039414f2a
#owner: https://api.github.com/users/AngieEspinosa97

import pytest
import requests
from main import *

url= "https://api.stagingeb.com/v1/properties"
apikey  =   "l7u502p8v46ba3ppgvj5y2aad50lb9"
headers =   {'Content-Type': 'application/json', 'X-Authorization': apikey}

def test_set_url():
    response = requests.get(url, headers=headers)
    assert response.status_code == 200

def test_get_pages():
    response = requests.get(url, headers=headers)
    response_body = response.json()
    assert response_body ["pagination"]["page"] == 1

def test_get_all_properties():
    response = requests.get(url, headers=headers)
    response_body = response.json()
    assert response_body ["content"][7]["title"] == "Oficinas en Venta en Valle Oriente"


if __name__ == "__main__":
    test_set_url()

if __name__ == "__main__":
    test_get_pages()

if __name__ == "__main__":
    test_get_all_properties()