#date: 2022-04-12T16:52:40Z
#url: https://api.github.com/gists/2aba13dd47750de00ef3aab6c37f4eca
#owner: https://api.github.com/users/worbas

#!/usr/bin/env python3
"""
TOTP DEMO

Reference: https://github.com/pyauth/pyotp

Dipendenze:

 $ pip3 install pyotp
 $ apt install qrencode

"""

import pyotp
import subprocess
import time

# Token segreto
B32S = pyotp.random_base32()

# Parametri
cifre = 6
secondi = 30

# TOTP Object
totp = pyotp.totp.TOTP(B32S, digits=cifre, interval=secondi)

# Provisioning URI
PU = totp.provisioning_uri(name="totp@example.com", issuer_name="EXAMPLE Inc.")

print("\n*** TOTP DEMO ***\n")

print("Secret =", B32S, "\n") 

print("Provisioning URI =", PU, "\n")

# QR Code
subprocess.run(("qrencode", "-t", "UTF8", PU))

print()

try:
    while True:
        print(time.asctime(), "TOTP =", totp.now())
        time.sleep(5)
except KeyboardInterrupt:
    print('\r Game Over!')

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4