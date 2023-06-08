#date: 2023-06-08T16:54:07Z
#url: https://api.github.com/gists/0331b5283b9eae0c61dc06af695d761d
#owner: https://api.github.com/users/faruquesarker

import requests
import json

GITLAB_API_ENDPOINT = 'https://gitlab.com/api/v4/version'

def get_gitlab_version(headers):
    """
    Retrieves Gitlab version by calling GitLab API with an access token in the request header 

    parameters:
    url: Gitlab URL
    headers: "**********"
    """
    try:
        if headers:
            resp = requests.get(GITLAB_API_ENDPOINT, headers=headers)
            print(f"GitLab returns status code: {resp.status_code}")
        else: 
            raise(f"Missing headers for auth. Supply a header with PRIVATE_TOKEN")
        
        if resp.status_code == 200:
            content = json.loads(resp.content)
            version = content['version']
            print(f"Got Gitlab version: {version}")
        else:
            raise(f"Other response code: {resp.status_code}")
    except Exception as e:
        raise(f"Failed request with error: {e}") 


if __name__ == '__main__':
    token = input("Enter your access token: "**********"
    headers = { 'PRIVATE-TOKEN': "**********"
    get_gitlab_version(headers=headers)