#date: 2022-01-03T17:03:00Z
#url: https://api.github.com/gists/7005cead15f2106250c23245bd74c09c
#owner: https://api.github.com/users/LDAP

#!/bin/python3
# author: Lucas Alber

# Script to control pipewire using wpctl (Wireplumber) from polybar.
# Dependencies: ONLY pipewire with wireplumber.
# Commands:
#   up, down: Volume control
#   mute-{sink,source}: Toggle mute of current default
#   next-{sink,source}: Make next device default

# [module/pipewire]
# type = custom/script
# tail = true
# format = <label>
# ; label-font = 2
# exec = ~/dotfiles/scripts/wpctl-simple.py
# click-right = ~/dotfiles/scripts/wpctl-simple.py next-sink &
# click-left = ~/dotfiles/scripts/wpctl-simple.py mute-sink &
# scroll-up = ~/dotfiles/scripts/wpctl-simple.py up &
# scroll-down = ~/dotfiles/scripts/wpctl-simple.py down &

import re
import sys
import os
import asyncio
from datetime import timedelta, datetime
from math import ceil
from typing import List
from dataclasses import dataclass

PRINT_SYMBOL = True
PRINT_PERCENTAGE = True
PRINT_SINK_NAME = False
VOLUME_STEP = .1
# After an update block monitoring for the specified time
MONITORING_BLOCKING = 50  # milliseconds
# Wireplumber takes a moment to update
MONITORING_DELAY = 100 # milliseconds

@dataclass
class Entry:
    id: int
    name: str
    volume: float
    is_default: bool
    is_muted: bool


@dataclass
class Status:
    audio_sinks: List[Entry]
    audio_sources: List[Entry]


async def get_status() -> Status:
    p = await asyncio.create_subprocess_shell("wpctl status", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    stdout, stderr = await p.communicate()

    status = Status([], [])

    av = ""
    cat = ""
    for line in stdout.decode().splitlines():
        if line == "Audio" or line == "Video":
            av = line

        category = re.search(r".*├─\s+([A-Za-z]+):", line)
        if category:
            cat = category.group(1)

        r = re.match(r"[^\*]*(\*)?\s+(\d+)\.\s+(.*)\s+\[vol: ([^\s]*)\s*(MUTED)?\]", line)
        if r:
            is_default = r.group(1) is not None
            is_muted = r.group(5) is not None
            id = int(r.group(2))
            name = r.group(3).strip()
            volume = float(r.group(4))
            match av:
                case "Audio":
                    match cat:
                        case "Sinks":
                            status.audio_sinks += [Entry(id, name, volume, is_default, is_muted)]
                        case "Sources":
                            status.audio_sources += [Entry(id, name, volume, is_default, is_muted)]
                case "Video":
                    pass
    return status

def get_volume_symbol(entry: Entry):
    if entry.is_muted:
        return "ﱝ"
    symbols = ["", "", "", ""]
    return symbols[ceil(entry.volume * (len(symbols) - 1))]

def get_polybar_repr(entry: Entry) -> str:
    repr = []
    if PRINT_SYMBOL:
        repr += [get_volume_symbol(entry)]
    if PRINT_PERCENTAGE:
        repr += [f"{100 * entry.volume:.01f}%"]
    if PRINT_SINK_NAME:
        repr += [entry.name]

    return " ".join(repr)

def set_volume(entry: Entry, vol: float):
    vol = max(min(vol, 1), 0)
    os.system(f"wpctl set-volume {entry.id} {vol}")

def toggle_mute(entry: Entry):
    os.system(f"wpctl set-mute {entry.id} toggle")

def next_entry(entries: List[Entry]):
    default_index = [i for i,x in enumerate(entries) if x.is_default][0]
    new_default_index = (default_index + 1) % len(entries)

    os.system(f"wpctl set-default {entries[new_default_index].id}")


async def handle_pw_mon_event(line: str, last_handled: datetime) -> datetime:
    if not datetime.now() - last_handled >= timedelta(milliseconds=MONITORING_BLOCKING):
        return last_handled

    await asyncio.sleep(MONITORING_DELAY / 1000)
    status = await get_status()
    default_sink = [x for x in status.audio_sinks if x.is_default][0]
    default_source = [x for x in status.audio_sources if x.is_default][0]

    print(get_polybar_repr(default_sink), flush=True)

    return datetime.now()


async def start_pipewire_monitoring():
    p = await asyncio.create_subprocess_exec(
            *[
                "pw-mon",
                "--no-colors"
            ],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

    last_event = datetime.min
    while True:
        # For some reason pw-mon prints to stderr?
        data = await p.stderr.readline()  # type: ignore
        if not data:
            return
        line = data.decode('utf-8').rstrip()
        
        last_event = await handle_pw_mon_event(line, last_event)

async def main():
    if len(sys.argv) == 1:
        monitoring = asyncio.create_task(start_pipewire_monitoring())
        await monitoring

    else:
        status = await get_status()
        default_sink = [x for x in status.audio_sinks if x.is_default][0]
        default_source = [x for x in status.audio_sources if x.is_default][0]

        match sys.argv[1]:
            case "up":
                set_volume(default_sink, default_sink.volume + VOLUME_STEP)
            case "down":
                set_volume(default_sink, default_sink.volume - VOLUME_STEP)
            case "mute-sink":
                toggle_mute(default_sink)
            case "mute-source":
                toggle_mute(default_source)
            case "is-sink-mute":
                print(int(default_sink.is_muted))
            case "is-source-mute":
                print(int(default_source.is_muted))
            case "next-sink":
                next_entry(status.audio_sinks)
            case "next-source":
                next_entry(status.audio_sources)

if __name__ == "__main__":
    asyncio.run(main())
