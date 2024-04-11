#date: 2024-04-11T16:57:35Z
#url: https://api.github.com/gists/a67be97955c5760dc1527659922df28e
#owner: https://api.github.com/users/NateM135

import discord
import nyt_parse

from discord import app_commands
from typing import Dict, Union

# Configure these accordingly
MY_GUILD = discord.Object(id=0)
OWNER_USERID = 0

class MyClient(discord.Client):
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        self.tree.copy_global_to(guild=MY_GUILD)
        # await self.tree.sync()

class GameRecord():
    def __init__(self, user_id: int, time: int) -> None:
        self.user_id = user_id
        self.time = time
    
    def getTime(self) -> int:
        return self.time
    
    def getUserID(self) -> int:
        return self.user_id

intents = discord.Intents.default()
intents.message_content = True
client = MyClient(intents=intents)

def addEmojiIfGoodPlacement(placement: int):
    if placement==1:
        return f"🥇 {placement}"
    if placement==2:
        return f"🥈 {placement}"
    if placement==3:
        return f"🥉 {placement}"
    return f"{placement}"

async def get_scoreboard_text(record_date: str, user_id: int) -> str:

    if record_date not in client.game_data:
        return None

    records = sorted(client.game_data[record_date], key=lambda r: int(r.getTime()))

    user_position = None
    user_record = None
    for i, record in enumerate(records):
        if record.getUserID() == user_id:
            user_position = i + 1
            user_record = record
            break

    scoreboard_message = f"** {record_date} Leaderboard**\n```"
    for i, record in enumerate(records, start=1):
        print(record.getTime())
        user = await client.fetch_user(record.getUserID())
        if user:
            username = f"@{user.name}"
        else:
            username = f"User (ID: {record.getUserID()})"
        scoreboard_message += f"{addEmojiIfGoodPlacement(i)}. {username}: {record.getTime()} seconds\n"
        if len(scoreboard_message) >= 1800:
            break

    if user_position:
        scoreboard_message += f"```\nYou are currently in position {user_position} on the leaderboard with a time of {user_record.getTime()} seconds!"

    return scoreboard_message

async def addUserPlay(user_id: int, data: Dict[str, Union[str, int]]) -> bool:
    if "date" not in data or "time" not in data:
        return False
    date_of_score = data["date"]
    if date_of_score not in client.game_data:
        client.game_data[date_of_score] = []
    else:
        for record in client.game_data[date_of_score]:
            if record.getUserID() == user_id:
                return False
    play = GameRecord(user_id, data["time"])
    client.game_data[date_of_score].append(play)
    return True


@client.event
async def on_ready():
    print(f'Logged in as {client.user} (ID: {client.user.id})')
    print('------')
    client.game_data = {}


@client.tree.command()
async def nyt_ping(interaction: discord.Interaction):
    # sanity check command
    await interaction.response.send_message(f'Pong =)')

@client.tree.command()
async def sync(ctx):
    print("sync command")
    if ctx.author.id == OWNER_USERID:
        await client.tree.sync()
        await ctx.send('Command tree synced.')
    else:
        await ctx.send('You must be the owner to use this command!')

@client.tree.command()
@app_commands.describe(
    user='User to attribute the shared play to',
    link='Link for the play',
)
async def direct_add_link(interaction: discord.Interaction, user: Union[discord.Member, discord.User], link: str):
    if interaction.user.id != OWNER_USERID:
        return await interaction.response.send_message("Command can only be used by owner.", ephemeral=True)
    if nyt_parse.validate_nyt_mini_url(link):
        data = nyt_parse.get_date_and_time_strings(link)
        res = await addUserPlay(user.id, data)
        if res:
            text = await get_scoreboard_text(data["date"], user.id)
            return await interaction.response.send_message(text)
        else:
            return await interaction.response.send_message("Failed to add play to record. User may already have play added for today.")
    await interaction.response.send_message("Link is invalid.", ephemeral=True)


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if nyt_parse.validate_nyt_mini_url(message.content):
        data = nyt_parse.get_date_and_time_strings(message.content)
        res = await addUserPlay(message.author.id, data)
        if res:
            text = await get_scoreboard_text(data["date"], message.author.id)
            await message.channel.send(text)
        else:
            await message.channel.send("Something went wrong when adding this play to the leaderboard.")

client.run('TOKEN')