#date: 2023-03-06T16:53:04Z
#url: https://api.github.com/gists/8fd1006326cc49b26691f71c3b15d46d
#owner: https://api.github.com/users/smitmartijn

import asyncio
from Smile import Smile
import json

async def main():
        smile = "**********"
        await smile.connect()
        await smile.update_locations()

        # Get the current power consumption in watts
        current_power = smile.get_power_data_from_location("location-id")
        print(json.dumps(current_power))

        await smile.close_connection()

if __name__ ==  '__main__':
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())til_complete(main())