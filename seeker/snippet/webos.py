#date: 2022-01-17T17:02:40Z
#url: https://api.github.com/gists/32061701981846bd8efbed43163dbbb7
#owner: https://api.github.com/users/abates

import logging

import sys
import asyncio
from bscpylgtv import WebOsClient

HOST="theater-tv"

fmt = "%(asctime)s %(levelname)s (%(threadName)s) [%(name)s] %(message)s"
datefmt = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(format=fmt, datefmt=datefmt, stream = sys.stdout, level = logging.INFO)

logging.getLogger("bscpylgtv.webos_client").setLevel(logging.DEBUG)
_LOGGER = logging.getLogger(__name__)

class Listener:
    def __init__(self):
        _LOGGER.info("Setting up host variables")
        self.host=HOST
        self.config_file="./webostv.conf"
        self.client=None

    async def async_handle_state_update(self, client):
        _LOGGER.info(f"State update {self.client.power_state}")

    
    async def run(self):
        self.client = await WebOsClient.create('theater-tv')
        while True:
            if self.client.is_connected():
                await asyncio.sleep(5)
            elif self.client.connect_task is None or self.client.connect_task.done():
                try:
                    _LOGGER.info("Connecting")
                    await self.client.connect()
                    await self.client.register_state_update_callback(self.async_handle_state_update)
                    _LOGGER.info("Connected!")
                except asyncio.TimeoutError  as e:
                    _LOGGER.info(f"Failed to connect: TimeoutError")
                    await asyncio.sleep(5)
                except Exception as e:
                    _LOGGER.info(f"Failed to connect: {type(e)}")
                    await asyncio.sleep(5)

f = Listener()
asyncio.run(f.run())
