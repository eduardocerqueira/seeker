#date: 2024-06-07T16:47:55Z
#url: https://api.github.com/gists/78d9b5eb4a0ee11f5e8c0322fcba6b94
#owner: https://api.github.com/users/dongreenberg

from concurrent.futures import ThreadPoolExecutor
import time

from bs4 import BeautifulSoup
import requests
import runhouse as rh


def word_counts(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    words = soup.get_text().split()
    return {word: words.count(word) for word in words}


if __name__ == "__main__":
    NUM_NODES = 4
    cluster = rh.cluster(
        f"rh-{NUM_NODES}x2CPU",
        instance_type="CPU:2",
        num_instances=NUM_NODES,
        den_auth=True,
        spot=True,
    ).up_if_not()

    NUM_REPLICAS = 8
    workers = []
    for i in range(NUM_REPLICAS):
        env = rh.env(
            name=f"count_env_{i}",
            reqs=["bs4", "requests"],
            compute={"CPU": 1},
        )
        worker_fn = rh.function(word_counts).get_or_to(cluster, env=env, name=f"word_counts_{i}")
        workers.append(worker_fn)

    urls = [
        "https://en.wikipedia.org/wiki/Python_(genus)",
        "https://en.wikipedia.org/wiki/Python_(mythology)",
        "https://en.wikipedia.org/wiki/Python_(painter)",
        "https://en.wikipedia.org/wiki/Python_(Efteling)",
        "https://en.wikipedia.org/wiki/Python_(automobile_maker)",
        "https://en.wikipedia.org/wiki/Python_(nuclear_primary)",
        "https://en.wikipedia.org/wiki/Python_(missile)",
        "https://en.wikipedia.org/wiki/Python_(codename)",
        "https://en.wikipedia.org/wiki/Python_(film)",
    ]

    # Call the replicas in parallel in a threadpool
    start = time.time()
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        def call_with_round_robin(*args):
            while not workers:
                time.sleep(.25)
            worker = workers.pop(0)
            result = worker(*args, stream_logs=False)
            workers.append(worker)
            return result
        all_counts = executor.map(call_with_round_robin, urls)
