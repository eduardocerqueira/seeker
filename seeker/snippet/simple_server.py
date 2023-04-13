#date: 2023-04-13T17:09:15Z
#url: https://api.github.com/gists/488ef95f177966899653ad770cbf4445
#owner: https://api.github.com/users/nicksemin

import socket
import threading

i = 0

def connection(c, addr):
    global i
    print(addr[0], "connected.")
    name = ""
    while True:
        data = c.recv(1024)
        print("Rx: " + data.decode())
        message = "Server msg {}".format(i)
        c.sendall(message.encode())
        i += 1
        print("Tx: " + message)

addr = ("", 2023)
if socket.has_dualstack_ipv6():
    s = socket.create_server(addr, family=socket.AF_INET6, dualstack_ipv6=True)
else:
    s = socket.create_server(addr)

online_users = {}

while True:
    c, addr = s.accept()
    threading.Thread(target=connection, args=(c, addr)).start()

s.close()