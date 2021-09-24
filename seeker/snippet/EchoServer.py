#date: 2021-09-24T17:11:57Z
#url: https://api.github.com/gists/b8cbf0a6e3878f90d17bf3cf7b22cd83
#owner: https://api.github.com/users/tspyrou

# Echo server program
import socket

HOST = ''                 # Symbolic name meaning all available interfaces
PORT = 50007              # Arbitrary non-privileged port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)

conn, addr = s.accept()
print 'Connected by', addr
while 1:
  data = conn.recv(1024)
  if not data: break
  conn.sendall(data)
conn.close()
