#date: 2024-06-21T16:49:59Z
#url: https://api.github.com/gists/5eecd7516102065f4ad28d247ec7bd21
#owner: https://api.github.com/users/vndee

import threading
import time

def io_bound_task():
    time.sleep(2)
    print("I/O operation complete")

threads = []
for _ in range(5):
    thread = threading.Thread(target=io_bound_task)
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()
