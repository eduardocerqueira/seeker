#date: 2022-03-24T17:08:38Z
#url: https://api.github.com/gists/515c74dcaf47a52a32ebb5a95b331db5
#owner: https://api.github.com/users/ms1995

import os
import sys
import concurrent.futures
import threading
import subprocess

# Runs each command on a single, dedicated CPU.
# Useful when `isolcpus` is enabled and need to run many single-threaded programs.
CPUS = list(range(12, 24)) + list(range(36, 48))
WORKER_THREAD_PREFIX = 'pWorker'

with open(sys.argv[1], 'r') as f:
    cmds = f.readlines()

def worker(cmd):
    n = threading.current_thread().name
    if not n.startswith(WORKER_THREAD_PREFIX + '_'):
        print('[ERROR] Thread name is not prefixed with {}'.format(WORKER_THREAD_PREFIX))
        return
    n = int(n[len(WORKER_THREAD_PREFIX + '_'):])
    c = CPUS[n]
    print('[INFO] Running on CPU #{}: {}'.format(c, cmd))
    subprocess.run(['taskset', '-c', str(c), 'bash', '-c', cmd])

executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(CPUS), thread_name_prefix=WORKER_THREAD_PREFIX)
for i, cmd in enumerate(cmds):
    print('[INFO] Submitting command #{}, total {}'.format(i + 1, len(cmds)))
    executor.submit(worker, cmd.strip())