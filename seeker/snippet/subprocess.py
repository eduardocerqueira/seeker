#date: 2025-12-02T16:51:59Z
#url: https://api.github.com/gists/baa8132ae5215ffe6c5112a9ae90c6c1
#owner: https://api.github.com/users/pr0gramista

import subprocess
import threading
import time
import os

p = subprocess.Popen(
    ["bash"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding="utf-8",
)

try:
    out, err = p.communicate("ls\n", 15)
    print(out)
except TimeoutError:
    p.kill()
    out, err = p.communicate()


# Alternative

# data = ""
# def reader():
#     global data
#     while True:
#         data += p.stdout.readline()


# reader = threading.Thread(target=reader)
# reader.start()

# p.stdin.write("ls\n")
# p.stdin.flush()

# time.sleep(1)
# print(data)
