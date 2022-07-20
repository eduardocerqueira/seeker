#date: 2022-07-20T17:08:01Z
#url: https://api.github.com/gists/f36be6afd9cf8d96f0c049efd886761e
#owner: https://api.github.com/users/aiven-amartin

import requests
import sys
from time import sleep

headers = {"Content-Type": "application/json", "Accept": "application/json"}


def reindex(url, source, dest) -> bool:
    print(f"Reindexing {source} -> {dest}")
    body = {
        "conflicts": "proceed",
        "source": {"index": source}, 
        "dest": {"index": dest, "op_type": "create"}
    }
    response = requests.post(f"{url}/_reindex?wait_for_completion=false", headers=headers, json=body)
    if response.status_code != 200:
        print(f"FAILED to reindex {response.status_code}: {response.text}")
        return False
    task = response.json()["task"]
    print(f"Waiting for task {task} to complete")
    while True:
        try:
            response = requests.get(f"{url}/_tasks/{task}", headers=headers)
        except Exception:  # timeouts and stuff
            sleep(1)
            continue
        if response.status_code == 200:
            if response.json().get("completed"):
                print(f"Task {task} completed\n{response.text}")
                break
        elif response == 404:  # does not exist, assume to have completed
            print(f"Task {task} does not exist, assume completed")
            break
        else:
            print(f"unexpected result {response.status_code}: {response.text}")
            return False
        sleep(10)
    return True


def wait_until_green(url, idx) -> None:
    print(f"Waiting for {idx} to go green")
    while True:
        try:
            resp = requests.get(f"{url}/_cat/indices/{idx}", headers=headers)
            if resp.status_code == 404:
                raise RuntimeError(f"Index {idx} does not exist")
            health = resp.json()[0].get("health")
        except Exception:
            sleep(1)
            continue
        if health == "green":
            return
        sleep(10)


def get_indices(url, suffix) -> list[str]:
    response = requests.get(f"{url}/_cat/indices/*{suffix}", headers=headers)
    if response.status_code != 200:
        print(f"FAILED to retrieve indices {response.status_code}: {response.text}")
        return [] 
    return [idx['index'] for idx in response.json()]


if __name__ == "__main__":
    suffix = sys.argv[2] # "_aiven-restored"
    suffix_len = len(suffix)
    url = sys.argv[1]
    for idx in get_indices(url=url, suffix=suffix):
        print(f"Processing {idx}")
        dest = idx[:-suffix_len]
        if reindex(url, idx, dest):
            print(f"Processed {idx}, waiting for green")
            wait_until_green(url=url, idx=dest)
            print(f"{idx} updated and green")
