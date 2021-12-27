#date: 2021-12-27T17:02:02Z
#url: https://api.github.com/gists/9a1b669d5957f0ffd2c2806d162644e4
#owner: https://api.github.com/users/Kqpa

# Get started by creating a project in Replit

import os, discord
from discord.ext import commands

bot = commands.Bot(command_prefix="!!")

@bot.event
async def on_connect():
    await bot.change_presence(status=discord.Status.online,
                              activity=discord.Game('This is the bots activity status.'))
    print("Bot is online.")


@bot.command()
async def hello(ctx):
    await ctx.send("Hey")

bot.run(os.getenv('TOKEN'))
my_secret = os.environ['TOKEN']
# create a secret named 'TOKEN' and put your bot's token to the value