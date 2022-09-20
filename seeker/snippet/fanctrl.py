#date: 2022-09-20T17:05:33Z
#url: https://api.github.com/gists/cac084fb0a0148582744d225ab2ba483
#owner: https://api.github.com/users/soimy

# Do the imports
import os
import sys
# OPi Library and instructions: https://pypi.org/project/OrangePi.GPIO/
import OPi.GPIO as GPIO
import time
from datetime import datetime
# from time import sleep
# Orange Pi Setup
# GPIO.setboard(GPIO.ZERO)
GPIO.setmode(GPIO.SUNXI)
GPIO.setwarnings(False)
# Targeted temperature to start the fan (40 degrees celsius)
startTemp = 50000
stopTemp = 35000
# Im using GPIO and a PNP transistor to control the fan, by default it will be enabled. 1 sets the GPIO to HIGH and 0 to LOW.
outputValue = 0
# Output GPIO pin for the fan transistor. Check the pinout at: https://i0.wp.com/oshlab.com/wp-content/uploads/2016/11/Orange-Pi-Zero-Pinout-banner2.jpg?fit=1200%2C628&ssl=1
outputPin = 'PA6' 
GPIO.setup(outputPin, GPIO.OUT)

interval = 10
# Reads the armbian temperature file and checks if it's higher than the threshold, if positive, turns the fan on, otherwise, off.
while True:
  with open("/etc/armbianmonitor/datasources/soctemp", 'r') as fin:
    temp = 0 
    temp = int(fin.read())
    if (temp >= startTemp and outputValue == 0):
      outputValue = 1
      print("[" + datetime.now().strftime("%H:%M:%S") + "] Fan run at temp: " + str(temp/1000))
      GPIO.output(outputPin, outputValue)
    elif (temp < stopTemp and outputValue == 1):
      outputValue = 0
      print("[" + datetime.now().strftime("%H:%M:%S") + "] Fan stop at temp: " + str(temp/1000))
      GPIO.output(outputPin, outputValue)
  # GPIO.cleanup()
  time.sleep(interval)

