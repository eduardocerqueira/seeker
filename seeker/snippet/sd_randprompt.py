#date: 2024-01-03T17:04:43Z
#url: https://api.github.com/gists/e3edf80bbf61cf6a971160a242eb3bff
#owner: https://api.github.com/users/KatsumiKougen

#!/usr/bin/env python3

import os, struct, subprocess, random
#import icecream as ic

def GenRndNumber(begin, end, rounding=0):
    Number = struct.unpack("Q", os.urandom(8))[0] / 2 ** 64
    RangeDist = float(end) - float(begin)
    if rounding == 0:
        return float(begin) + Number * RangeDist
    else:
        return round(float(begin) + Number * RangeDist, rounding)

Scheduler = ("DDIM", "DDPM", "PNDM", "LMS", "HEUN", "EULER", "EULERA", "DPMM",
             "KDPM2", "KDPM2A", "DEIS", "UNIPC", "DPM_SDE")

Rand32 = lambda: subprocess.check_output(
    "/home/katsumi/rand.sh 32",
    shell=True, executable="/bin/sh"
).decode("utf-8").strip()

PosPrompt = " ".join([Rand32() for i in range(32)])
NegPrompt = " ".join([Rand32() for i in range(32)])

print(
    "Go to \x1b[3;93mhttps://mage.space\x1b[0m and enter the following options:\n"
    "\n"
    f"Positive prompt: \x1b[1m{PosPrompt}\x1b[0m\n"
    "\n"
    f"Negative prompt: \x1b[1m{NegPrompt}\x1b[0m\n"
    "\n"
    f"Guidance scale: \x1b[1;94m{GenRndNumber(0, 30, 1)}\x1b[0m "
    f"| Refiner strength: \x1b[1;94m{GenRndNumber(0, 0.6, 2)}\x1b[0m\n"
    "\n"
    f"Scheduler: {random.choice(Scheduler)}"