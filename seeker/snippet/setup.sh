#date: 2025-11-04T17:08:38Z
#url: https://api.github.com/gists/1217b45127946c35c9da8de8729a4f45
#owner: https://api.github.com/users/UniversPlayer

#!/usr/bin/env bash

set -e

echo "=== Discord Music Bot Setup ==="

read -s -p "Enter your Discord Bot Token (hidden): "**********"

echo

TARGET_DIR="discord_music_bot"

mkdir -p "$TARGET_DIR"

cd "$TARGET_DIR"

cat > requirements.txt <<'PY'

discord.py==2.4.1

yt-dlp

python-dotenv

PY

cat > .env <<EOF

DISCORD_TOKEN= "**********"

EOF

cat > music_utils.py <<'PY'

import yt_dlp

YTDL_OPTIONS = {

    "format": "bestaudio/best",

    "noplaylist": True,

    "quiet": True,

    "no_warnings": True,

    # Use ytsearch to find top result when query is not a url

    "default_search": "ytsearch",

    "skip_download": True,

    "extract_flat": False,

}

def yt_search(query: str):

    """Return a dict with keys: url (direct media URL for ffmpeg), title, webpage_url"""

    with yt_dlp.YoutubeDL(YTDL_OPTIONS) as ytdl:

        info = ytdl.extract_info(query, download=False)

        # If default_search=ytsearch, result comes as 'entries' with first entry

        if 'entries' in info and info['entries']:

            entry = info['entries'][0]

        else:

            entry = info

        # yt-dlp provides a direct URL for streaming in 'url' for some extractors

        # For safety, ask yt-dlp to prepare a format URL

        # We will return the 'url' field if it exists, otherwise 'webpage_url'

        stream_url = entry.get('url') or entry.get('webpage_url')

        return {

            "title": entry.get("title", "Unknown title"),

            "webpage_url": entry.get("webpage_url"),

            "stream_url": stream_url

        }

PY

cat > bot.py <<'PY'

import os

import json

import asyncio

from dotenv import load_dotenv

import discord

from discord.ext import commands

from discord import app_commands

from music_utils import yt_search

load_dotenv()

TOKEN = "**********"

 "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"T "**********"O "**********"K "**********"E "**********"N "**********": "**********"

    raise SystemExit("No DISCORD_TOKEN found in .env")

PREFIX = ","

intents = discord.Intents.default()

intents.message_content = True

intents.voice_states = True

bot = commands.Bot(command_prefix=PREFIX, intents=intents, help_command=None)

FAVS_FILE = "favs.json"

file_lock = asyncio.Lock()

def safe_load_favs():

    if not os.path.exists(FAVS_FILE):

        return {}

    with open(FAVS_FILE, "r", encoding="utf-8") as f:

        return json.load(f)

async def safe_save_favs(data):

    async with file_lock:

        with open(FAVS_FILE, "w", encoding="utf-8") as f:

            json.dump(data, f, ensure_ascii=False, indent=2)

class ControlView(discord.ui.View):

    def __init__(self, ctx, voice_client):

        super().__init__(timeout=None)  # persistent for this runtime

        self.ctx = ctx

        self.voice_client = voice_client

    @discord.ui.button(label="Stop", style=discord.ButtonStyle.danger)

    async def stop_button(self, interaction: discord.Interaction, button: discord.ui.Button):

        if interaction.user != self.ctx.author:

            await interaction.response.send_message("Only the command user can control this.", ephemeral=True)

            return

        if self.voice_client and self.voice_client.is_playing():

            self.voice_client.stop()

        try:

            await self.voice_client.disconnect()

        except:

            pass

        await interaction.response.edit_message(content="⏹ Stopped and disconnected.", view=None)

    @discord.ui.button(label="Start again", style=discord.ButtonStyle.success)

    async def start_button(self, interaction: discord.Interaction, button: discord.ui.Button):

        if interaction.user != self.ctx.author:

            await interaction.response.send_message("Only the command user can control this.", ephemeral=True)

            return

        # attempt to resume if paused, otherwise do nothing

        if self.voice_client:

            if self.voice_client.is_paused():

                self.voice_client.resume()

                await interaction.response.edit_message(content="▶ Resumed playback.", view=self)

                return

            else:

                await interaction.response.send_message("Nothing to resume. Use ,play <song> to start a new song.", ephemeral=True)

                return

        await interaction.response.send_message("No active voice client.", ephemeral=True)

@bot.event

async def on_ready():

    print(f"✅ Logged in as {bot.user} (id: {bot.user.id})")

@bot.command(name="help")

async def help_cmd(ctx):

    text = (

        "**Music Bot Commands**\n"

        "`{p}help` — Show this help message\n"

        "`{p}active [#voice-channel]` — Make the bot join the voice channel (or your VC if omitted)\n"

        "`{p}play <song name or url>` — Search YouTube and play the top result\n"

        "`{p}stop` — Stop playing and disconnect\n"

        "`{p}addfav <song name or url>` — Add a song to your favourites\n"

        "`{p}favourites` — Show your favourite songs\n"

        "\nWhile a song plays, a message with **Stop** and **Start again** buttons appears — only the command user can use them."

    ).format(p=PREFIX)

    await ctx.send(text)

@bot.command(name="active")

async def active(ctx, channel: discord.VoiceChannel = None):

    # Join the specified voice channel, or the user's voice channel if omitted

    target = channel

    if target is None:

        if not ctx.author.voice or not ctx.author.voice.channel:

            await ctx.reply("You are not in a voice channel and none specified.")

            return

        target = ctx.author.voice.channel

    try:

        await target.connect()

        await ctx.reply(f"✅ Joined **{target.name}**")

    except Exception as e:

        # If already connected, move to that channel

        vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)

        if vc:

            await vc.move_to(target)

            await ctx.reply(f"✅ Moved to **{target.name}**")

        else:

            await ctx.reply(f"❌ Could not join: {e}")

@bot.command(name="play")

async def play(ctx, *, query: str):

    # Make sure user is in voice channel

    if not ctx.author.voice or not ctx.author.voice.channel:

        await ctx.reply("You must be in a voice channel to use ,play.")

        return

    channel = ctx.author.voice.channel

    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)

    if not vc or not vc.is_connected():

        vc = await channel.connect()

    else:

        if vc.channel != channel:

            await vc.move_to(channel)

    await ctx.trigger_typing()

    try:

        info = yt_search(query)

    except Exception as e:

        await ctx.reply(f"❌ Error while searching: {e}")

        return

    title = info.get("title", "Unknown")

    stream_url = info.get("stream_url") or info.get("webpage_url")

    if not stream_url:

        await ctx.reply("❌ Couldn't get a playable URL for that query.")

        return

    # FFmpeg options to improve reconnecting for streams

    before_opts = "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5"

    source = discord.FFmpegPCMAudio(stream_url, before_options=before_opts, executable="ffmpeg")

    # Use a volume transformer in case user wants to change later

    player = discord.PCMVolumeTransformer(source, volume=1.0)

    # stop any current playing audio

    if vc.is_playing():

        vc.stop()

    # play and set an after callback to disconnect

    def _after(err):

        if err:

            print("Player error:", err)

        # try to disconnect after playback finishes

        coro = disconnect_after(ctx)

        try:

            asyncio.run_coroutine_threadsafe(coro, bot.loop)

        except Exception as e:

            print("Failed to schedule disconnect:", e)

    vc.play(player, after=_after)

    view = ControlView(ctx, vc)

    msg = await ctx.send(f"▶ Playing: **{title}**", view=view)

    # no return value needed

async def disconnect_after(ctx, delay: int = 1):

    await asyncio.sleep(delay)

    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)

    if vc and not vc.is_playing() and not vc.is_paused():

        try:

            await vc.disconnect()

        except:

            pass

@bot.command(name="stop")

async def stop_cmd(ctx):

    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)

    if not vc or not vc.is_connected():

        await ctx.reply("🔕 I'm not connected to a voice channel.")

        return

    if vc.is_playing() or vc.is_paused():

        vc.stop()

    try:

        await vc.disconnect()

    except:

        pass

    await ctx.reply("⏹ Stopped and disconnected.")

@bot.command(name="addfav")

async def addfav(ctx, *, query: str):

    await ctx.trigger_typing()

    try:

        info = yt_search(query)

    except Exception as e:

        await ctx.reply(f"❌ Search error: {e}")

        return

    title = info.get("title", "Unknown")

    webpage = info.get("webpage_url")

    user_id = str(ctx.author.id)

    data = safe_load_favs()

    user_list = data.get(user_id, [])

    # store minimal info

    user_list.append({"title": title, "webpage": webpage})

    data[user_id] = user_list

    await safe_save_favs(data)

    await ctx.reply(f"✅ Added to favourites: **{title}**")

@bot.command(name="favourites")

async def favourites(ctx):

    data = safe_load_favs()

    user_list = data.get(str(ctx.author.id), [])

    if not user_list:

        await ctx.reply("You have no favourites. Add one with `,addfav <song>`.")

        return

    lines = []

    for i, item in enumerate(user_list, start=1):

        t = item.get("title", "Unknown")

        web = item.get("webpage") or ""

        if web:

            lines.append(f"{i}. {t} — <{web}>")

        else:

            lines.append(f"{i}. {t}")

    # send as a code block if long

    await ctx.send("**Your favourites:**\n" + "\n".join(lines))

# Run bot

if __name__ == "__main__":

    bot.run(TOKEN)

PY

cat > README.md <<'MD'

# Discord Music Bot

Commands (prefix `,`):

- `,help` — Show commands

- `,active [#voice-channel]` — Bot joins a voice channel (or your voice channel if omitted)

- `,play <song name or url>` — Search YouTube and play the top result

- `,stop` — Stop playing & disconnect

- `,addfav <song name or url>` — Add the top search result to your favourites

- `,favourites` — Show your saved favourites

Requirements:

- Python 3.9+

- ffmpeg available on PATH

- A Discord bot token stored in `.env`

To run:

1. Make sure ffmpeg is installed on your machine.

2. `python3 -m pip install -r requirements.txt`

3. `python3 bot.py`

MD

echo "Installing Python packages..."

python3 -m pip install -r requirements.txt

# Try install ffmpeg on Debian/Ubuntu (if permitted)

if command -v apt-get >/dev/null 2>&1; then

  echo "Trying to install ffmpeg via apt-get (may require sudo)..."

  if [ "$(id -u)" -eq 0 ]; then

    apt-get update && apt-get install -y ffmpeg || echo "apt install failed; please install ffmpeg manually."

  else

    echo "Not running as root. Skipping automated ffmpeg install. If ffmpeg missing, install it manually (apt, brew, choco, etc)."

  fi

fi

echo "Setup complete. Starting bot..."

python3 bot.py