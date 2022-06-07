#date: 2022-06-07T17:06:30Z
#url: https://api.github.com/gists/2c20a9e0d4e7ca7da2497d818bb6e368
#owner: https://api.github.com/users/majek

from socket import SOCK_STREAM, AF_INET, SOL_SOCKET, SO_REUSEADDR, SOL_TCP, TCP_MAXSEG, SO_RCVBUF, SO_SNDBUF, IPPROTO_IP, TCP_NODELAY, TCP_QUICKACK
import ctypes
import os
import socket
import struct
import sys
import time
import threading
import select

fast = sys.argv[1] == 'fast'

sd = socket.socket(AF_INET, SOCK_STREAM, 0)
sd.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
sd.setsockopt(SOL_TCP, TCP_MAXSEG, 1024)
sd.setsockopt(SOL_SOCKET, SO_RCVBUF, 16*1024*1024)
sd.bind(('127.0.0.3', 1234))
sd.listen(32)

cd = socket.socket(AF_INET, SOCK_STREAM, 0)
cd.setsockopt(SOL_SOCKET, SO_SNDBUF, 4*1024*1024)

cd.bind(('127.0.0.2', 0))
cd.connect(('127.0.0.3', 1234))

# move to ESTAB state ASAP, since tcp_rcv_established is for ESTAB indeed
ssd, _ = sd.accept()

done = False

waitinsec = 0.05
def thread_fill():
    burst_payload = b'a'* (1*1024*1024)

    cd.setblocking(False)
    while not done:
        try:
            cd.send(burst_payload)
            if fast:
                ssd.setsockopt(SOL_TCP, TCP_QUICKACK, 1)
            select.select([], [cd], [], waitinsec / 1000)
        except BlockingIOError:
            time.sleep(waitinsec)

def thread_drain():
    bytespersec = 12500000

    bytesperiter = int(bytespersec * waitinsec)
    ssd.setblocking(False)
    offset = 0.
    while not done:
        wanted = waitinsec - offset
        t0 = time.time()
        try:
            ssd.recv(bytesperiter)
        except BlockingIOError:
            pass
        time.sleep(wanted)
        td = time.time()-t0
        offset = td - wanted


fill_thrd = threading.Thread(target=thread_fill, args=())
fill_thrd.start()
drain_thrd = threading.Thread(target=thread_drain, args=())
drain_thrd.start()
try:
    time.sleep(1.5)
except KeyboardInterrupt:
    print("Ctrl+C pressed...")
done = True
fill_thrd.join()
drain_thrd.join()
os.system('ss -memoi sport = :1234 or dport = :1234|cat')
