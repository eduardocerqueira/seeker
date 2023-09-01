#date: 2023-09-01T16:51:04Z
#url: https://api.github.com/gists/7fa5e8a3f7234f9e2aa75874f5a6f97e
#owner: https://api.github.com/users/antunesleo

from time import sleep
import requests


UNSTABLE_API = "https://httpbin.org/status/200,500"


def notsodumbretry(max_attempts, retry_backoff, backoff_exponential):
    succeeded = False
    attempts = 0
    
    while attempts <= max_attempts:
        response = requests.get(UNSTABLE_API)
        succeeded = response.ok
        attempts += 1
        
        if succeeded:
            break
            
        sleep(retry_backoff)
        if backoff_exponential:
            retry_backoff = retry_backoff * 2

    if succeeded:
        print(f"succeded after {attempts} attempts")
    else:
        print(f"failed after {attempts} attemps")


if __name__ == "__main__":
    notsodumbretry(max_attempts=3, retry_backoff=4, backoff_exponential=True)
