#date: 2025-02-18T16:47:13Z
#url: https://api.github.com/gists/42f98944ed7866c1d66d4366d16bf3da
#owner: https://api.github.com/users/dellybot

import delpy
import logging

from discord.ext import commands, tasks
from logging.handlers import RotatingFileHandler

logger = logging.getLogger('del.py')
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler(
    filename='path/to/prefered/log_dir/del.log',
    encoding='utf-8',
    mode='w',
    maxBytes=10 * 1024 * 1024,  # 10 MB
    backupCount=5,
    delay=0
)
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(handler)

class discordextremelist(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.update_stats.start()
        # you can get the bot's token from your bot's page on DEL
        self.delapi = "**********"
        self.delapi.start_loop(wait_for=1800) # wait_for argument is optional and defaults at 1800.

    def cog_unload(self):
        self.delapi.close_loop()

def setup(bot):
    bot.add_cog(discordextremelist(bot))ot.add_cog(discordextremelist(bot))