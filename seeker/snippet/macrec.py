#date: 2022-01-06T17:08:39Z
#url: https://api.github.com/gists/a02cd3f0cec1b997eabf851ec0d59537
#owner: https://api.github.com/users/LinusCDE

#!/usr/bin/env python3

from threading import Thread, Lock
import evdev
from time import time

startedAt = time()
lock = Lock()

def rec(evNum):
    global lastEventAt
    device = evdev.InputDevice('/dev/input/event%d' % evNum)
    for event in device.read_loop():
        if event.type == 1 and event.code == 1 and event.value == 1:
            # Esc pressed
            exit(0)
        with lock:
            print("Event %d %f %d %d %d" % (evNum, event.timestamp() - startedAt, event.type, event.code, event.value))


# Change these ids to what evdev ids you want recorded!!!
Thread(target=rec, args=(8,), daemon=True).start()
rec(10) # Last id can just be on the current thread
