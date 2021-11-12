#date: 2021-11-12T17:08:09Z
#url: https://api.github.com/gists/87cfebe847f39030fd305058f96363ca
#owner: https://api.github.com/users/Foolson

#!/usr/bin/env python3

# Python modules
import time
import sys
import signal
import getopt
from rpi_hardware_pwm import HardwarePWM

# Configuration
waitTime = 5     # [s] Time to wait between each refresh
pwmFrequency = 25000

# Configurable temperature and fan speed
minTemp = 44
maxTemp = 80
fanLow = 15
fanHigh = 100
fanOff = 0
fanMax = 100

# Logging and metrics (enable = 1)
verbose = 0
quiet = 0

# Parse input arguments
helpText = ('pi-pwm-fan.py [--min-temp=40] [--max-temp=70] [--fan-low=20] '
            '[--fan-high=100] [--wait-time=1] [-v|--verbose] '
            '[-q|--quiet] [-h|--help]')

try:
    opts, args = getopt.getopt(sys.argv[1:], "hvq", [
        "min-temp=", "max-temp=", "fan-low=", "fan-high=", "wait-time=",
        "help", "verbose", "quiet"
    ])
except getopt.GetoptError:
    print(helpText)
    sys.exit(2)

for opt, arg in opts:
    if opt in ("-h", "--help"):
        print(helpText)
        sys.exit()
    elif opt in ("-v", "--verbose"):
        verbose = 1
    elif opt in ("-q", "--quiet"):
        quiet = 1
    elif opt in ("--min-temp"):
        minTemp = int(arg)
    elif opt in ("--max-temp"):
        maxTemp = int(arg)
    elif opt in ("--fan-low"):
        fanLow = int(arg)
    elif opt in ("--fan-high"):
        fanHigh = int(arg)
    elif opt in ("--wait-time"):
        waitTime = int(arg)

if quiet == 0:
    print("Min temp:", minTemp)
    print("Max temp:", maxTemp)
    print("Fan low:", fanLow)
    print("Fan high:", fanHigh)
    print("Wait time:", waitTime)
    print("Verbose:", verbose)
    print("Quiet:", quiet)


# Get CPU's temperature
def getCpuTemperature():
    with open('/sys/devices/virtual/thermal/thermal_zone0/temp') as f:
        lines = f.read()
    temp = (float(lines)/1000)
    return temp


# Set fan speed
def setFanSpeed(pwm, speed):
    pwm.change_duty_cycle(speed)


# Get fan speed
def getFanSpeed(temp):
    # Turn off the fan if temperature is below minTemp
    if temp < minTemp:
        speed = fanOff

    # Set fan speed to MAXIMUM if the temperature is above maxTemp
    elif temp > maxTemp:
        speed = fanMax

    # Caculate dynamic fan speed
    else:
        step = (fanHigh - fanLow)/(maxTemp - minTemp)
        delta = temp - minTemp
        speed = fanLow + (round(delta) * step)

    return speed


# Stop the PWM fan and exit the script
def exitCleanup(signum, frame):
    pwm.change_duty_cycle(0)
    pwm.stop()
    print("PWM fan is turned off")
    sys.exit(signum)


# Setup GPIO pin for PWM
pwm = HardwarePWM(0, hz=pwmFrequency)
pwm.start(0)

# Catch signals to gracefully stop the fans if killed
signal.signal(signal.SIGINT, exitCleanup)
signal.signal(signal.SIGTERM, exitCleanup)

# Manage fan speed every waitTime sec
try:
    while True:
        temp = getCpuTemperature()
        speed = getFanSpeed(temp)
        setFanSpeed(pwm, speed)
        if verbose == 1 and quiet == 0:
            print("fan speed: ", int(speed), "    temp: ", temp)
        time.sleep(waitTime)
except KeyboardInterrupt:
    exitCleanup()
