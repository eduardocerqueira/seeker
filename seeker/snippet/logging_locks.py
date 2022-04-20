#date: 2022-04-20T17:00:27Z
#url: https://api.github.com/gists/1e2e4cb8f9fd35032fb56358f93754ec
#owner: https://api.github.com/users/clayg

import logging
import eventlet.patcher
eventlet.patcher.monkey_patch(thread=True)
import threading


def take_and_release():
    try:
        logging._lock.acquire()
    finally:
        logging._lock.release()

assert logging._lock.acquire()
t = threading.Thread(target=take_and_release)
t.start()
t.join(timeout=1)
# we should timeout, and the thread is still blocking
assert t.isAlive()
logging._lock.release()
t.join(timeout=1)
assert not t.isAlive()
print('locking works')
