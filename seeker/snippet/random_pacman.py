#date: 2021-12-24T16:36:48Z
#url: https://api.github.com/gists/386bf5728bd65c7846162a0965ee47e0
#owner: https://api.github.com/users/uidops

#!/usr/bin/python3
import subprocess
import random
import os

cows = [
    "ghost",
    "radio",
    "pod",
    "udder",
    "milk",
    "whale",
    "sheep",
    "dragon",
    "homer",
    "fox",
    "wizard",
    "taxi",
    "blowfish.cow"


]

_ = [1,0]
__ = random.choice(_)

if __ == 1:
    packages = subprocess.check_output("pacman -Q | cut -d ' ' -f1", shell=True)
    packages = packages.decode().split("\n")
    packages.pop()

    random_package = random.choice(packages)
    data = subprocess.check_output('pacman -Qi {} | grep "Description" | cut -d ":" -f2'.format(random_package), shell=True)

    data = "{}:{}".format(random_package, data.decode())
else:
    data = subprocess.check_output("fortune", shell=True).decode()

data = data.replace("\"", "\\\"")
os.system("cowsay -f {} \"{}\"".format(random.choice(cows), data))
