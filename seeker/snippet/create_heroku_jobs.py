#date: 2022-11-08T17:06:37Z
#url: https://api.github.com/gists/1d22c9f592fa34aa51be5c3d6afa09db
#owner: https://api.github.com/users/xoelop

import json

import requests
from dotenv import load_dotenv

load_dotenv(override=True)
import os

for i in range(3, 52):

    print(i)

    url = f"https://particleboard.heroku.com/apps/{os.getenv('HEROKU_APP_ID')}/jobs"

    command = 'your command here'

    payload = json.dumps(
        {
            "data": {
                "attributes": {
                    "at": 0,
                    "every": 10, # 10 - every 10 minutes, 60, every hour, 1440, every day. If 60, set at to the minutes. If 1440, set at to minutes since 00:00
                    "ran-at": None,
                    "create-at": None,
                    "updated-at": None,
                    "command": command,
                    "dyno-size": "Hobby",
                },
                "type": "jobs",
            }
        }
    )
    headers = {
        "authority": "particleboard.heroku.com",
        "accept": "application/vnd.api+json",
        "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
        "authorization": f"Bearer {os.getenv('HEROKU_API_KEY')}",
        "content-type": "application/vnd.api+json",
        "origin": "https://dashboard.heroku.com",
        "referer": "https://dashboard.heroku.com/",
        "sec-ch-ua": '"Chromium";v="106", "Google Chrome";v="106", "Not;A=Brand";v="99"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36",
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    response.raise_for_status()
