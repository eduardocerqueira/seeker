#date: 2022-04-27T17:09:58Z
#url: https://api.github.com/gists/813009bea87e545e7d0e171a886796b7
#owner: https://api.github.com/users/sidd-kishan

import socket
import time 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('0.0.0.0', 8585 ))
s.listen(0)                 
 
while True:
    client, addr = s.accept()
    client.settimeout(5)
    while True:
        content = client.recv(1024)
        if len(content) ==0:
           break
        if str(content,'utf-8') == '\r\n':
            continue
        else:
            print(str(content,'utf-8'))
            client.send(b'Hello From Python')
    client.close()