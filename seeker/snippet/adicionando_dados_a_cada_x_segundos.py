#date: 2024-05-03T16:54:43Z
#url: https://api.github.com/gists/2587012b2cd506bed708e8c734a78bef
#owner: https://api.github.com/users/fabioafreitas

import requests, random
from datetime import datetime
import time

if __name__ == '__main__':
    tempo_segundos = 10
    device_access_token = "**********"

    while True:
        dt_now = int(datetime.timestamp(datetime.now())*1000)
        fake_telemetry = {"ts":dt_now, "values":{"temperature":random.uniform(25, 30),"ph":random.uniform(5, 8)}}
        res = requests.post(
            url = f'https: "**********"
            headers={
                "Content-Type":"application/json"
            },
            json=fake_telemetry
        )
        print(res.status_code)
        time.sleep(tempo_segundos)print(res.status_code)
        time.sleep(tempo_segundos)