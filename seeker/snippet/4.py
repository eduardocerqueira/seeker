#date: 2023-12-11T16:49:07Z
#url: https://api.github.com/gists/1368364e2bf18eaea8be1cbbff1aed35
#owner: https://api.github.com/users/vvozian



import threading

THREAD_COUNT = 4
TASK_SIZE = 2500
N = THREAD_COUNT * TASK_SIZE
sum = 0
lock = threading.Lock()
tls = threading.local()

def compute(start, end):
    thread_sum = 0
    for i in range(start, end + 1):
        tls.counter += 1
        thread_sum += i
    return thread_sum

def worker(start, end):
    global sum
    tls.counter = 0
    thread_sum = compute(start, end)
    with lock:
        sum += thread_sum
    print(f"Thread {threading.get_ident()} counter:({tls.counter}): Computed sum from {start} to {end} = {thread_sum}")

threads = []
for i in range(THREAD_COUNT):
    start = i * TASK_SIZE + 1
    end = (i + 1) * TASK_SIZE
    t = threading.Thread(target=worker, args=(start, end))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print(f"Parent: Sum of numbers from 1 to {N} = {sum}")