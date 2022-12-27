#date: 2022-12-27T16:47:03Z
#url: https://api.github.com/gists/ab798c87197d54d46c24bc2b5e51320f
#owner: https://api.github.com/users/danieldean

#!/usr/bin/python3

# 
# stats_and_fan
# 
# Copyright (c) 2020 Daniel Dean <dd@danieldean.uk>.
# 
# Licensed under The MIT License a copy of which you should have 
# received. If not, see:
# 
# http://opensource.org/licenses/MIT
# 

import time
import netifaces
import psutil
from board import SCL, SDA
import busio
from PIL import Image, ImageDraw, ImageFont
import adafruit_ssd1306
import threading
import signal
from gpiozero import OutputDevice

# Set the display size here.
SCREEN_WIDTH = 128
SCREEN_HEIGHT = 32

# Set fan GPIO
FAN_GPIO = 16

# Set fan on and off thresholds.
ON_THRESHOLD = 45
OFF_THRESHOLD = 35


def signal_handler(signum, frame):
    global running
    running.set()


def get_ip():
    try:
        return netifaces.ifaddresses('br0')[netifaces.AF_INET][0]['addr']
    except (KeyError, ValueError):
        pass
    try:
        return netifaces.ifaddresses('eth0')[netifaces.AF_INET][0]['addr']
    except KeyError:
        pass
    try:
        return netifaces.ifaddresses('wlan0')[netifaces.AF_INET][0]['addr']
    except KeyError:
        return 'Disconnected'


def get_cpu():
    temperature = round(psutil.sensors_temperatures()['cpu_thermal'][0].current, 1)
    return psutil.cpu_percent(), temperature


def get_memory():
    memory = psutil.virtual_memory()
    used = round(memory.used / 1048576)
    total = round(memory.total / 1048576)
    percent = round(used / total * 100)
    return used, total, percent


def get_disk():
    disk = psutil.disk_usage('/')
    used = round(disk.used / 1073741824)
    total = round(disk.total / 1073741824)
    return used, total, disk.percent


def main():

    # To handle shutdowns and clear the screen.
    global running
    running = threading.Event()
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Initialise fan.
    fan = OutputDevice(FAN_GPIO)

    # Initialise and clear display.
    i2c = busio.I2C(SCL, SDA)
    disp = adafruit_ssd1306.SSD1306_I2C(SCREEN_WIDTH, SCREEN_HEIGHT, i2c)
    disp.fill(0)
    disp.show()

    # Create blank image for drawing.
    width = disp.width
    height = disp.height
    image = Image.new("1", (width, height))

    # Get drawing object to draw on image.
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)

    # Set starting coordinates.
    y = -2
    x = 0

    # Set font to default.
    font = ImageFont.load_default()

    # Update until SIGINT or SIGTERM.
    while not running.is_set():

        # Draw a black filled box to clear the image.
        draw.rectangle((0, 0, width, height), outline=0, fill=0)

        # Get the values.
        ip = get_ip()
        cpu_percent, cpu_temperature = get_cpu()
        memory_used, memory_total, memory_percent = get_memory()
        disk_used, disk_total, disk_percent = get_disk()

        # Control the fan based on CPU temperature.
        if cpu_temperature > ON_THRESHOLD and not fan.value:
            fan.on()
        elif cpu_temperature < OFF_THRESHOLD and fan.value:
            fan.off()

        # Draw stats.
        draw.text((x, y + 0), 'IP: ' + ip, font=font, fill=255)
        draw.text((x, y + 8), 'CPU: ' + str(cpu_percent) + '% ' + str(cpu_temperature) + '°C', font=font, fill=255)
        draw.text((x, y + 16), 'RAM: ' + str(memory_used) + '/' + str(memory_total) + 'MB ' + str(memory_percent) + '%', font=font, fill=255)
        draw.text((x, y + 25), 'Disk: ' + str(disk_used) + '/' + str(disk_total) + 'GB ' + str(disk_percent) + '%', font=font, fill=255)

        # Display image.
        disp.image(image)
        disp.show()
        time.sleep(1)

    # Clear display.
    disp.fill(0)
    disp.show()


if __name__ == "__main__":
    main()
