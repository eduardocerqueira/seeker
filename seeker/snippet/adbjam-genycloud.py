#date: 2022-01-10T16:57:57Z
#url: https://api.github.com/gists/96acd92d8dc823a342e752c582a59b97
#owner: https://api.github.com/users/d4vidi

#!/usr/bin/env python3

import os
import sys
import signal
import datetime

def log(text):
	now=datetime.datetime.now().strftime("%H:%M:%S.%f")
	print("@"+now+" <"+ANDROID_SERIAL+"> " + text)

def exec(attempt, command):
	log('[EXEC #'+attempt+'] ' + command)
	os.system(command)


###

class TimeOutException(Exception):
   pass

def alarm_handler(signum, frame):
    log("Timeout!")
    raise TimeOutException()

signal.signal(signal.SIGALRM, alarm_handler)

###

ANDROID_SERIAL=os.getenv('ANDROID_SERIAL')
APK_PATH='<todo>'
PACKAGE_NAME='<todo>'

if not ANDROID_SERIAL:
	raise Exception("Error: You must set the ANDROID_SERIAL env-var (e.g. ANDROID_SERIAL=localhost:12345)")
 
def adbinstall(attempt):
	exec(attempt, 'adb uninstall '+PACKAGE_NAME)
	exec(attempt, 'adb push '+APK_PATH+' /data/local/tmp/detox/Application.apk')
	exec(attempt, 'adb shell pm install -r -g -t /data/local/tmp/detox/Application.apk')
	# exec(attempt, 'adb install --no-streaming '+APK_PATH)

for i in range(10):
	signal.alarm(240)
	adbinstall(str(i))
	signal.alarm(0)
