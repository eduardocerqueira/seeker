#date: 2022-05-31T17:00:25Z
#url: https://api.github.com/gists/324ca109f91a37d5c3e62a808406f31a
#owner: https://api.github.com/users/RomainBrault

#!/usr/bin/env python

import asyncio
import random


TIME = 10


async def sort(x, max_arr=1):
    await asyncio.sleep(TIME * x / max_arr)
    print(x)


async def main():
    max_arr = 10000
    len_arr = 1000
    arr_unsorted = (random.randint(0, max_arr) for _ in range(len_arr))
    await asyncio.gather(
        *(sort(i, max_arr) for i in arr_unsorted)
    )


if __name__ == "__main__":
    asyncio.run(main())