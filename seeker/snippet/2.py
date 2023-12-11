#date: 2023-12-11T16:49:07Z
#url: https://api.github.com/gists/1368364e2bf18eaea8be1cbbff1aed35
#owner: https://api.github.com/users/vvozian

import threading
import signal
import sys

THREAD_COUNT = 4
TASK_SIZE = 250000000
CONSISTENCY_DELTA = 1000
N = THREAD_COUNT * TASK_SIZE
sum = 0
lock = threading.Lock()

exit_flag = False
cancellation_type = sys.argv[1]
if (cancellation_type != 'async' and cancellation_type != 'deffered'):
    raise Exception(f"Unknown cancellation type: {cancellation_type}")

def is_cancellation_allowed(start, value):
    global cancellation_type
    if cancellation_type == 'async':
        return True
    elif cancellation_type == 'deffered':
        return (value - start) % CONSISTENCY_DELTA == 0
    else:
        raise Exception(f"Unknown cancellation type: {cancellation_type}")

def compute(start, end):
    global exit_flag
    thread_sum = 0
    real_end = start
    for i in range(start, end + 1):
        if exit_flag and is_cancellation_allowed(start, i):
            break
        thread_sum += i
        real_end+=1
    return {
        "start": start,
        "end": real_end,
        "sum": thread_sum
    }

def worker(start, end):
    global sum
    thread_sum = compute(start, end)
    with lock:
        sum += thread_sum["sum"]
    print(f"Thread {threading.get_ident()}: Computed sum from {thread_sum['start']} to {thread_sum['end']} = {thread_sum['sum']}")

def sigint_handler(signum, frame):
    global exit_flag
    print(f"Received signal {signum}. Setting exit flag.")
    exit_flag = True

signal.signal(signal.SIGINT, sigint_handler)

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
