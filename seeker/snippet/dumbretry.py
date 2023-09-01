#date: 2023-09-01T16:50:24Z
#url: https://api.github.com/gists/98cfd8011ed9b1c69e3209dc0c861c98
#owner: https://api.github.com/users/antunesleo

import requests

UNSTABLE_API = "https://httpbin.org/status/200,500,503,401"


def dumbretry():
    succeeded = False
    attempts = 0
    
    while not succeeded:
        response = requests.get(UNSTABLE_API)
        succeeded = response.ok
        attempts += 1
    
    print(f"succeded after {attempts} attempts")


if __name__ == "__main__":
    dumbretry()
