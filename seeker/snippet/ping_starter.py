#date: 2024-11-19T16:46:18Z
#url: https://api.github.com/gists/fa02f816478b9ce36250c8f2d623db02
#owner: https://api.github.com/users/dulimarta

from socket import gethostbyname, getprotobyname, socket, SOCK_RAW, AF_INET, htons, inet_ntoa
from os import getpid
import sys
from struct import pack, unpack_from
from time import time, sleep
from select import select

ICMP_ECHO_REQUEST = 8

def checksum(sdata):
  csum = 0
  
  # Pad with zero if we have odd number of bytes
  if len(sdata) % 2:
    data += b'\x00'
    
  for i in range(0, len(sdata), 2):
    # Take each 16-bit as an unsigned integer
    word = (sdata[i] << 8) + sdata[i+1]
    csum += word
  # Add back the carry value to the current sum
  csum = (csum >> 16) + (csum & 0xFFFF)
  # Just in case the above sum has carry bits
  csum = csum + (csum >> 16)
  
  # Return the lowest 16 bits of the 1s complement   
  return ~csum & 0xFFFF

def receiveOnePing(mySocket, ID, timeout, destAddr):
  timeLeft = timeout
  while True:
    startedSelect = time()
    whatReady = select([mySocket], [], [], timeout)
    howLongInSelect = time()  - startedSelect
    if whatReady[0] == []:
      timeLeft = timeLeft - howLongInSelect
      if timeLeft <= 0:
        print ("Timeout")
        return None   # timeout
    else:
      break
  timeReceived = time()
  recPacket,addr = mySocket.recvfrom(1024)
  ## BEGIN: your code
  ## Fetch the IP header fields
  ## Fetch the ICMP header fields
  print ("Your work here")    
  ## END: your code
  return 0

def sendOnePing(mySocket, destAddr, ID):
  
  # ICMP header fields: 
  #    type      (1 byte) 
  #    code      (1 byte)
  #    checksum  (2 bytes)
  #    id        (2 bytes)
  #    sequence  (2 bytes)
  
  # Make a dummy header with 0 checksum
  myChecksum = 0
  header = pack("bbHHH", ICMP_ECHO_REQUEST, 0, myChecksum, ID, 1)
  data = pack("d", time())
  # calculate the checksum on the header and dummy data
  myChecksum = checksum(header+data)
  header = pack("bbHHH", ICMP_ECHO_REQUEST, 0, htons(myChecksum), ID, 1)
  packet = header+data  
  mySocket.sendto(packet, (destAddr, 1))

def doOnePing(destAddr, timeout):
  icmp = getprotobyname("icmp")
  mySocket = socket(AF_INET, SOCK_RAW, icmp)
  # Use the lowest 16-bit of the PID
  myID = getpid() & 0xFFFF
  sendOnePing(mySocket, destAddr, myID)
  rtt = receiveOnePing(mySocket, myID, timeout, destAddr)
  mySocket.close()
  return rtt

def ping(host, timeout=1):
  try:
    dest = gethostbyname(host)
    print (f"Pinging {host}")
    while True:
      doOnePing(dest, timeout)
      sleep(1)
  except Exception as e:
    print(f"Exception: {e}")
  except KeyboardInterrupt:
    print("\nShow summary here")

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print(f"Use {sys.argv[0]} hostname")
  else:
    ping(sys.argv[1])
