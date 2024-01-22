#date: 2024-01-22T16:44:17Z
#url: https://api.github.com/gists/631d10f28eee4f05de71fcc917581450
#owner: https://api.github.com/users/coderbyheart

#!/usr/bin/env python3 

import os 
from _thread import * 
import socket 
import datetime 

localIP     = "0.0.0.0" 
localPort   = 2444 
bufferSize  = 1024 
msgFromServer       = "Hello UDP Client" 
bytesToSend         = str.encode(msgFromServer) 

# Create a datagram socket 
UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM) 

# Bind to address and ip 
UDPServerSocket.bind((localIP, localPort)) 

print("UDP server up and listening") 

def multi_threaded_client(message,address): 
    global UDPServerSocket 
    e = datetime.datetime.now() 
    res = b'Time: '+str.encode(e.strftime("%Y-%m-%d %H:%M:%S"))+b' Message: '+ str.encode(message.decode('utf-8')) 
    UDPServerSocket.sendto(res, address) 

# Listen for incoming datagrams 
while(True): 
    bytesAddressPair = UDPServerSocket.recvfrom(bufferSize) 
    message = bytesAddressPair[0] 
    address = bytesAddressPair[1] 
    clientMsg = "Message from Client:{}".format(message) 
    clientIP  = "Client IP Address:{}".format(address) 
    print(clientMsg) 
    print(clientIP) 
    # Sending a reply to client 
    start_new_thread(multi_threaded_client, (message,address )) 