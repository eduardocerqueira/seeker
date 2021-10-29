#date: 2021-10-29T17:00:41Z
#url: https://api.github.com/gists/2e4fcfd2bb59e5f32c46fd0cbd0a0bff
#owner: https://api.github.com/users/mdrakiburrahman

import requests
import json
from termcolor import colored

def aadtoken(client_id, client_secret, client_tenant):
    url = "https://login.microsoftonline.com/{}/oauth2/token".format(client_tenant)
    payload='grant_type=client_credentials&client_id={}&client_secret={}&resource=https%3A%2F%2Fpurview.azure.net'.format(client_id, client_secret)
    response = requests.request("POST", url, data=payload)

    return json.loads(response.text)['access_token']

def get_collection(access_token, collections_name):
    url = "https://{}.purview.azure.com/collections/{}?api-version=2019-11-01-preview".format(purview_account, collections_name)
    headers = {
        'Authorization': 'Bearer {}'.format(access_token),
    }
    response = requests.request("GET", url, headers=headers)
    return response.text

def delete_collection(access_token, collections_name):
    url = "https://{}.purview.azure.com/collections/{}?api-version=2019-11-01-preview".format(purview_account, collections_name)
    headers = {
        'Authorization': 'Bearer {}'.format(access_token),
    }
    response = requests.request("DELETE", url, headers=headers)
    return response.text

def create_collection(access_token, collections_name, collections_friendly_name, parent_collection_name):
    url = "https://{}.purview.azure.com/collections/{}?api-version=2019-11-01-preview".format(purview_account, collections_name)
    headers = {
        'Authorization': 'Bearer {}'.format(access_token),
        'Content-Type': 'application/json'
    }
    payload = json.dumps({
                "name": collections_name,
                "parentCollection": {
                    "type": "CollectionReference",
                    "referenceName": parent_collection_name
                },
                "friendlyName": collections_friendly_name
            })
    response = requests.request("PUT", url, headers=headers, data=payload)
    return response.text

def customprint(response):
    # If response contains "unauthorized" - print
    if "Unauthorized" in response:
        print(colored(response, 'red'))
        # raise Exception('Hit the authorization error!')
    else:
        print(colored(response, 'green'))


### MAIN LOOP ###
def runloop(access_token, purview_account):
    for i in range(1, 1000000):
        print('Round: {}'.format(i))

        for j in range(1, 30):
            collections_name = "TS{}{}".format(i,j)
            print('\nCollection Name: {}\n'.format(collections_name))
            
            # Create collection 
            print('\nCreate Collection\n')
            customprint(create_collection(access_token, collections_name, collections_name, purview_account))

            # Get collection
            print('\Get Collection\n')
            customprint(get_collection(access_token, collections_name))

            # Delete collection
            print('\Delete Collection\n')
            customprint(delete_collection(access_token, collections_name))

        print('\n\n\n')
    
    return null

if __name__ == "__main__":
    client_id = '....'
    client_secret = '....'
    client_tenant = '....'
    purview_account = '....'

    access_token = aadtoken(client_id, client_secret, client_tenant)
    
    try:
        runloop(access_token, purview_account)
    except Exception:
        pass


