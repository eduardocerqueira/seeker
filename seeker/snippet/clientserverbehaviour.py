#date: 2023-09-20T16:53:51Z
#url: https://api.github.com/gists/6015ebb0abf25c801277fa72cac77ef3
#owner: https://api.github.com/users/Sanju-Bhat

#client.py

from xmlrpc.client import ServerProxy
from_client = len(open("data.txt").read().split(" "))

if __name__ == '__main__':
    s = ServerProxy("http://172.17.15.9:3000")
    d = s.add(from_client)
    print(d)


#server.py
from xmlrpc.server import SimpleXMLRPCServer
import os
in_server = len(open("data.txt").read().split(" "))

def add(from_client):
    total = 0
    with open("result.txt", "r") as file:
            if os.path.getsize("result.txt") != 0:
                numbers = [int(line.strip()) for line in file.readlines()]
                total = sum(numbers) + from_client
                
            else:
                 total = from_client + in_server
    with open("result.txt", "w") as file: 
         file.write(str(total))           
    return in_server + from_client

if __name__ == "__main__":
    s = SimpleXMLRPCServer(("172.17.15.9", 3000))
    print("Server running")
    s.register_function(add)
    s.serve_forever()