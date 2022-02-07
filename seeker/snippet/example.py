#date: 2022-02-07T16:55:20Z
#url: https://api.github.com/gists/47bcc6b078abfbae44bafac3676cc01e
#owner: https://api.github.com/users/vicyap

from pprint import pprint

import ray
ray.init("ray://mycluster.internal:10001")

@ray.remote
def task():
    import time
    time.sleep(30)


pprint(ray.cluster_resources())
results = ray.get([task.remote() for _ in range(200)])
