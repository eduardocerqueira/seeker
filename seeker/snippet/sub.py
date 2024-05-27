#date: 2024-05-27T17:02:05Z
#url: https://api.github.com/gists/546a80b76ef97ffa749e91237e3ca92e
#owner: https://api.github.com/users/rdev32

import time
from datetime import datetime

def subscribe_for(days: int) -> None:
    now = time.time()
    format = datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S')
    print(f"You made your subscription in {format}")
    then = now + (86400 * days)
    format = datetime.fromtimestamp(then).strftime("%Y-%m-%d %H:%M:%S")
    print(f"Your subscription will end in {format}")


subscribe_for(30)
