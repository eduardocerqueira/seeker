#date: 2022-01-06T17:08:39Z
#url: https://api.github.com/gists/a02cd3f0cec1b997eabf851ec0d59537
#owner: https://api.github.com/users/LinusCDE

#!/usr/bin/env python3

import evdev
from time import sleep, time

def readcommands():
    '''Parse all input from stdin (until EOF / CTRL+D) and sort by time.'''
    commands = []
    while True:
        try:
            line = input()
        except EOFError:
            # Sort command by time (multiple recorded devices might not be exactly be in proper order)
            commands.sort(key=lambda cmd: cmd[2])
            return commands
        command = line.split()[0]
        args = line.split()[1:]
        if command == "Event":
            commands.append(("Event", int(args[0]), float(args[1]), int(args[2]), int(args[3]), int(args[4]) ))

commands = readcommands()
#print(commands)

# Find all evdev devices used in the recording and replicate them as new/cloned uinput devices
# (Maybe change to just inject into the orignal devices instead.)
devs = {}
for cmd in filter(lambda cmd: cmd[0] == "Event", commands):
    if cmd[1] not in devs:
        devs[cmd[1]] = evdev.UInput.from_device("/dev/input/event%d" % cmd[1])

#print(devs)
#print('Running...')

startedAt = time()

for cmd in commands:
    command = cmd[0]
    if command == "Event":
        # Wait until next event is supposed to run
        now = time() - startedAt
        while now < cmd[2]:
            sleep(max(0, cmd[2] - now))
            now = time() - startedAt
        # Send input
        devs[cmd[1]].write(cmd[3], cmd[4], cmd[5])

for dev in devs.values():
    dev.close()