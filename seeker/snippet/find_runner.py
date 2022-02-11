#date: 2022-02-11T16:41:28Z
#url: https://api.github.com/gists/a403956d1de7d6e7857b1488be0aadc1
#owner: https://api.github.com/users/felipevolpatto

#!/usr/bin/env python3

import psutil
import docker
from datetime import timedelta, datetime
import time

client = docker.from_env()
while True:
    time.sleep(10)
    try:
        c = client.containers.get('ephemeral-github-actions-runner.service')
    except docker.errors.NotFound as e:
        continue

    try:
        for UID, PID, PPID, C, STIME, TTY, TIME, CMD in c.top()['Processes']:
            if 'Runner.Worker' in CMD:
                p = psutil.Process(int(PID))
                elapsed = datetime.now() - datetime.fromtimestamp(p.create_time())
                if elapsed > timedelta(minutes=5):
                    print(f"[{datetime.now()}] Found runner job older than 5 minutes ({elapsed}): {UID} {PID} {PPID} {C} {STIME} {TTY} {TIME} {CMD}")
                    print("Killing...")
                    p.terminate()
    except docker.errors.APIError as e:
        continue