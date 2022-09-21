#date: 2022-09-21T17:22:59Z
#url: https://api.github.com/gists/f91de7e23c4488c071840b380efcb87f
#owner: https://api.github.com/users/liquid182

import subprocess
import re

POWER_MGMT_RE = re.compile('"DevicePowerState"=(\d)')

def display_status():
    output = subprocess.check_output(["/usr/sbin/ioreg","-w","0","-c","IODisplayWrangler","-r","IODisplayWrangler"])
    status = POWER_MGMT_RE.search(str(output)).group(1)
    return status

def isDisplayAwake():
    return int(display_status()) == 4

def isDisplayAsleep():
    return int(display_status()) == 1
