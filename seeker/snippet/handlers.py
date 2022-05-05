#date: 2022-05-05T16:54:14Z
#url: https://api.github.com/gists/72be157665f918f34d24ed70ba045f41
#owner: https://api.github.com/users/Zsailer

import json

from jupyter_server.extension.handler import ExtensionHandlerMixin
from jupyter_server.base.handlers import APIHandler
import tornado


import asyncio

async def doing_something():
    for i in range(100):
        await asyncio.sleep(1)


class RouteHandler(ExtensionHandlerMixin, APIHandler):
    # The following decorator should be present on all verb methods (head, get, post,
    # patch, put, delete, options) to ensure only authorized user can request the
    # Jupyter server

    @property
    def mytrait(self):
        return self.settings.get("mytrait")

    @tornado.web.authenticated
    async def get(self):
        task = asyncio.create_task(doing_something())
        self.settings["task"] = task
        self.finish(json.dumps({
            "data": f"This is /my_server_extension/get_example endpoint with the following trait: {self.mytrait}!"
        }))



class QueueHandler():

    async def get(self):
        task = self.settings["task"]
        await task