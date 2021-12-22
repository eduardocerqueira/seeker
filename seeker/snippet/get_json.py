#date: 2021-12-22T17:06:21Z
#url: https://api.github.com/gists/adef9a1d5b63919bd2e7e8616eae6883
#owner: https://api.github.com/users/megaleunam

import requests
import json

def get_json(url):
    """
        Retorna el json de la url pasada por parámetro
    """
    try:
        response = requests.get(url)
    except requests.exceptions.ConnectionError:
        return
    
    return json.dumps(response.json())

url = "https://ifconfig.co/json"
json_object = get_json(url)