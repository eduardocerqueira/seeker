#date: 2022-04-01T17:07:47Z
#url: https://api.github.com/gists/5427e567d4e02d8f0fac22c5a9f9118a
#owner: https://api.github.com/users/kensoi

from vkbotkit import librabot
from vkbotkit.objects import decorators, filters, enums, library_module
import asyncio
import os

send_hello_message = """
Hello, world
Programmed to work and not to feel
Not even sure that this is real
"""

class basic_lib(library_module):
    @decorators.callback(filters.whichUpdate({enums.events.message_new,}))
    async def send_hello(self, package):
        await package.toolkit.send_reply(package, send_hello_message)


async def main():
    bot = librabot(os.environ['VKBOTKIT_TOKEN'])

    # CONFIGURE LOGGER
    bot.toolkit.configure_logger(enums.log_level.DEBUG, True, True)

    # IMPORT LIBRARY
    bot.library.import_module(basic_lib)

    # START POLLING 
    await bot.start_polling()


loop = asyncio.new_event_loop()
loop.run_until_complete(main())