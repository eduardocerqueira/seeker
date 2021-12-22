#date: 2021-12-22T16:59:21Z
#url: https://api.github.com/gists/1a5ef9ba5113bf29ad4242a518e468ad
#owner: https://api.github.com/users/jagrafft

import board
import digitalio
import adafruit_max31865

from signal import SIGINT, signal
from subprocess import call
from sys import exit
from time import localtime, sleep, strftime
from os import mkdir

sample_rate = 28.9 #seconds, `fswebcam` takes approximately 1sec to execute
base_path = "/home/pi/monitor_data/starter_sessions"
date_fmt_display = "%Y-%m-%dT%H:%M:%S"
date_fmt_system = "%Y%m%dT%H%M%S"

spi = board.SPI()
cs = digitalio.DigitalInOut(board.D5)  # Chip select of the MAX31865 board.
sensor = adafruit_max31865.MAX31865(
  spi,
  cs,
  rtd_nominal=1000.0,
  ref_resistor=4300.0,
  wires=3
)

# Folders and files
session_file_path = f"{base_path}/{strftime('%Y-%m-%dT%H%M%S', localtime())}"
session_data_path = f"{session_file_path}/max31865_readings.csv"

mkdir(session_file_path)
session_data_file = open(session_data_path, "w", encoding="utf-8")
session_data_file.write("timestamp,C,Ω\n")

# File close hooks
def close(signum, frame):
    print('Closing...')
    print(f'signum: {signum}')
    print(f'frame: {frame}')
    exit()
    
signal(SIGINT, close)

# Sample from MAX31865
def sample(rate):
    while True:
        yield {
            'resistance': sensor.resistance,
            'temperature': sensor.temperature
        }
        sleep(rate)

# Data Capture Loop #
for temp in sample(sample_rate):
    ts = localtime()
    ts_str = strftime("%Y-%m-%dT%H:%M:%S", ts)

    cmd = [
        "fswebcam",
        "-d",
        "/dev/video0",
        "-i",
        "0",
        "-r",
        "1024x768",
        "--jpeg",
        "85",
        "--timestamp",
        ts_str,
        "--font",
        "arial:24",
        "--info",
        f"{temp['temperature']:.2f}C",
        f"{session_file_path}/{strftime('%Y-%m-%dT%H%M%S', ts)}.jpeg"
    ]

    subprocess_exit_code = call(cmd)

    data_row = ",".join([
        ts_str,
        f"{temp['temperature']}",
        f"{temp['resistance']}"
    ])
    
    session_data_file.writelines([data_row,"\n"])
    print(data_row,"OK" if subprocess_exit_code == 0 else "ERROR")
