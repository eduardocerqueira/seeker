#date: 2024-05-27T16:48:36Z
#url: https://api.github.com/gists/4543ec55f7a9498059ee0941451b3e48
#owner: https://api.github.com/users/PabloVD

#-----------------------------------
# Simple script to automatically find and kill defunct processes
# Author: Pablo Villanueva Domingo
#-----------------------------------

import os
import re 

# Find defunct processes and save them to temporary file
os.system("ps -ef | grep defunct > zombies.txt")

pids = []

# Load data from defunct processes file and remove the file
zombies = open("zombies.txt", "r").read()
os.system("rm zombies.txt")

print(zombies)

# For each zombie process, find the integers correspoding to the PID and PPID
for z in zombies.split(" "):
    ints = re.findall(r'^[-+]?([1-9]\d*|0)$',z)
    if len(ints)==1:
        pids.append(ints[0])

# There should be 3 integers per process
assert len(pids)%3==0

# Kill process by PID and PPID
for i in range(len(pids)//3):

    os.system("kill -9 "+pids[3*i]+" "+pids[3*i+1])

