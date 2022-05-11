#date: 2022-05-11T17:15:51Z
#url: https://api.github.com/gists/2b90989bd9c771c4cc7ac760317be0ff
#owner: https://api.github.com/users/lobotony

#!/usr/bin/env python3

import psutil
import time

def logDebug(msg):
	print(time.ctime() + " " + msg)

while True: 
	count = 0

	for p in psutil.process_iter():
		if "wdavdaemon_enterprise" == p.name():
			p.kill()
			p.wait()
			count = count +1

	logDebug("swatted: "+str(count))
	time.sleep(5)





