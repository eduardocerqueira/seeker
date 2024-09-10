#date: 2024-09-10T16:45:57Z
#url: https://api.github.com/gists/0150766320516a0cfcfda5d94c2d18fe
#owner: https://api.github.com/users/MattieOF

from telethon import *
import asyncio
import json
import os
import time
import dataclasses
import sys
import datetime
from dataclasses import dataclass

# Thanks to https://stackoverflow.com/a/54769644
def dataclass_from_dict(klass, d):
    try:
        fieldtypes = {f.name:f.type for f in dataclasses.fields(klass)}
        return klass(**{f:dataclass_from_dict(fieldtypes[f],d[f]) for f in d})
    except:
        return d # Not a dataclass field

api_id = 69420 # get your own at: https://my.telegram.org/apps
api_hash = "nuh uh, you're not getting this one :3"

client = TelegramClient('CuteBot', api_id, api_hash)

@dataclass
class ScheduledMessage:
    username: str
    message: str
    interval: int
    last_sent: float

scheduledMessages = []
if (os.path.exists("scheduledMessages.json")):
    with open("scheduledMessages.json", "r") as file:
        loadedMessages = json.load(file)
        scheduledMessages = [dataclass_from_dict(ScheduledMessage, msg) for msg in loadedMessages]

def save_messages():
    with open("scheduledMessages.json", "w") as file:
        scheduledMessagesAsDict = list(map(lambda msg: dataclasses.asdict(msg), scheduledMessages))
        json.dump(scheduledMessagesAsDict, file)

async def loop():
    logOut = False

    async def check_messages():
        didEdit = False
        for msg in scheduledMessages:
            if time.time() - msg.last_sent >= msg.interval:
                try:
                    await client.send_message(msg.username, msg.message)
                    msg.last_sent = time.time()
                    print(f"Sent message to {msg.username}! Next message in {msg.interval} seconds (at {datetime.datetime.fromtimestamp(time.time() + msg.interval).strftime('%Y-%m-%d %H:%M:%S')})")
                    didEdit = True
                except Exception as error:
                    print(f"Failed to send message to {msg.username}! Due to {error}")
        if didEdit:
            save_messages()

    if "autorun" in sys.argv:
        print("Running!")
        while True:
            await check_messages()
            await asyncio.sleep(1)

    while True:
        cmd = input("Enter command: ")
        if cmd == "exit":
            break
        elif cmd == "logout":
            logOut = True
            break
        
        cmd = cmd.split(" ")
        if cmd[0] == "list":
            # List all scheduled messages
            # Format: list
            if len(scheduledMessages) == 0:
                print("No scheduled messages yet :( Use the 'add' command to add one!")
                continue

            for i, msg in enumerate(scheduledMessages):
                print(f"{i}: To {msg.username}, \"{msg.message}\" (every {msg.interval} seconds)")
        elif cmd[0] == "add":
            # Add a scheduled message
            # Format: add <username> <interval in seconds> <message>
            if len(cmd) < 4:
                print("Not enough parameters! Format: add <username> <interval in seconds> <message>")
                continue
            
            try:
                interval = float(cmd[2])
                if interval < 1:
                    raise ValueError
            except ValueError:
                print("Invalid interval! Must be an integer above 1.")
                continue

            try:
                await client.send_message(cmd[1], "new scheduled message added :3")
            except:
                print("Invalid username! Make sure you have the correct username and that you've sent a message to the user at least once.")
                continue

            scheduledMessages.append(ScheduledMessage(username=cmd[1], interval=interval, message=" ".join(cmd[3:]), last_sent=time.time()))
            save_messages()
        elif cmd[0] == "remove":
            # Remove a scheduled message by index from the list
            # Format: remove <index>
            if len(cmd) < 2:
                print("Not enough parameters! Format: remove <index>")
                continue

            try:
                index = int(cmd[1])
                if index < 0 or index >= len(scheduledMessages):
                    raise ValueError
            except ValueError:
                print("Invalid index! Must be an integer within the range of the list.")
                continue

            scheduledMessages.pop(index)
            save_messages()
        elif cmd[0] == "run":
            while True:
                await check_messages()
                await asyncio.sleep(1)

    if logOut:
        await client.log_out()
        print("Logged out!")

with client:
    client.loop.run_until_complete(loop())

print("Bye!")
