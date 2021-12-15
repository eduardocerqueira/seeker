#date: 2021-12-15T16:59:58Z
#url: https://api.github.com/gists/d869c0e067696d247adbcaa5c638bdc5
#owner: https://api.github.com/users/Seanny123

import concurrent.futures
import random
import time

import ray

@ray.remote
def append_a(in_str):
    print(in_str)
    time.sleep(random.uniform(0, 1.5))
    return f"{in_str}a"


ray.init()

with concurrent.futures.ThreadPoolExecutor(max_workers=80) as threader:
    inputs = ["a", "b", "c", "d", "e", "f", "g"]
    outputs = []

    submitted = threader.map(append_a.remote, inputs)

    futures = {task.future(): inp for task, inp in zip(submitted, inputs)}

    for future in concurrent.futures.as_completed(futures):
        outputs.append((futures[future], future.result()))

print(outputs)
