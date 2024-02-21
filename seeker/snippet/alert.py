#date: 2024-02-21T17:02:23Z
#url: https://api.github.com/gists/0259814c64e9c40006f38254b2abcd45
#owner: https://api.github.com/users/mvandermeulen

#!/usr/bin/env python3
import sys
import requests
import redis
import socket
import pathlib
import site

teams_path = pathlib.Path(__file__).absolute().parent.parent.parent
site.addsitedir(teams_path)
from send_teams import send_es_alert # pylint: disable=import-error


def get_metrics():
    '''
    HTTP GET metrics from promethetheus endpoint
    Parse the results
    Return int value of dropped messages
    Return -1 if failure occured
    '''
    print(">>> start get_metrics()")

    # curl metrics endpoint
    endpoint = 'http://localhost:2112/metrics'
    try:
        req = requests.get(endpoint)
    except requests.exceptions.ConnectionError:
        print(f">>> Failed to HTTP GET from {endpoint}")
        return -1
    if req.status_code != 200:
        print(f">>> Failed to HTTP GET from {endpoint}")
        return -1

    # parse result
    text = req.text.split('\n')
    dropped_count_string = ""
    dropped_count = 0
    for i in text:
        if i.startswith('#'):
            continue
        if 'PromESDroppedMsg' in i:
            dropped_count_string = i
            dropped_count = dropped_count_string.split()[-1]
            dropped_count = int(dropped_count)
            break

    # return int value from parsed information
    print(">>> end get_metrics()")
    return dropped_count


def push_dropped_count_to_redis(count=0) -> int:
    '''
    Update redis key with current drop count
      compare the current count with previous and send alert with the difference , then update redis previous count with current
    '''
    
    redis_key = "SOC::GOLOGESD::DROPPEDMESSAGES"
    redis_cli = redis.Redis(host='localhost', port=6379)

    # if count is 0 exit, nothing to do
    if count == 0:
        print(">>> no dropped messages to report OK")

    # get value from redis
    previous_dropped_count = redis_cli.get(redis_key)
    if not previous_dropped_count:
        print(f">>> NOthing found in {previous_dropped_count}, setting value to {count}")
        # set the new value
        redis_cli.set(redis_key, count)
        return count

    print(f">>> previous count in redis is {previous_dropped_count}")
    previous_dropped_count = int(previous_dropped_count)

    # rare cases metric was reset, we should update redis with new (lower count)
    if count < previous_dropped_count:
        redis_cli.set(redis_key, count)
        print(f">>> Updated redis with new lower count {count}")
        return 0

    if count == previous_dropped_count:
        print(">>> No new dropped messages OK")
        return 0

    if count > previous_dropped_count:
        diff = count - previous_dropped_count
        # update redis with new value
        redis_cli.set(redis_key, count)
        return diff


def send_alert(count=0):
    '''
    Send message to teams garbo
    '''
    print(">>> starting send_alert()")
    hostname = socket.gethostname()
    send_es_alert.send_generic_alert(f'{hostname} dropped {count} messages')


def main():
    dropped_count = get_metrics()
    if dropped_count == -1:
        sys.exit(1)

    diff_count = push_dropped_count_to_redis(count=dropped_count)

    if diff_count and diff_count > 0:
        print(f"diff count is {diff_count}, sending alert to teams")
        send_alert(diff_count)
        return

    print(f">>> nothing to report, diff is {diff_count}")


if __name__ == '__main__':
    main()
