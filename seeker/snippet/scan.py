#date: 2022-10-27T17:18:41Z
#url: https://api.github.com/gists/0ba1a20b010d0320780f4df3dbbffa14
#owner: https://api.github.com/users/bschne

#!/opt/homebrew/bin/python3

import asyncio
from bleak import BleakScanner

async def main():
    devices = await BleakScanner.discover()
    for d in devices:
        print(d)

asyncio.run(main())
