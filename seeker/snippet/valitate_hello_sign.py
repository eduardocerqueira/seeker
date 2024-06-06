#date: 2024-06-06T16:46:33Z
#url: https://api.github.com/gists/c49094d451c225f1523474ca2bc84648
#owner: https://api.github.com/users/filipeandre

import requests
from requests.auth import HTTPBasicAuth

def validate_hellosign_credentials(api_key):
    url = 'https://api.hellosign.com/v3/account'

    response = requests.get(url, auth=HTTPBasicAuth(api_key, ''))

    if response.status_code == 200:
        print("Valid HelloSign credentials")
        account_info = response.json()
        print("Account Info:", account_info)
        return True
    else:
        print("Invalid HelloSign credentials")
        print("Response:", response.text)
        return False

if __name__ == "__main__":
    api_key = input("Enter your HelloSign API Key: ")
    validate_hellosign_credentials(api_key)
