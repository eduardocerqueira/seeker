#date: 2023-01-10T17:01:53Z
#url: https://api.github.com/gists/01864a123bdb437e26278d98229f2860
#owner: https://api.github.com/users/jabastien

"""
This example acts as a BLE HID keyboard to peer devices.
It get's keycode from UART RX.

Color indication:
* BLUE_LED is blinking when not connected and steady blue when connected
* NEOPIXEL alternate between RED / GREEN / BLUE every time a keycode is transmitted (up and down event)
"""

import time
import board
#from digitalio import DigitalInOut, Direction
import digitalio
import busio
import usb_hid

import adafruit_ble
from adafruit_ble.advertising import Advertisement
from adafruit_ble.advertising.standard import ProvideServicesAdvertisement
from adafruit_ble.services.standard.hid import HIDService
from adafruit_ble.services.standard.device_info import DeviceInfoService
from adafruit_hid.keyboard import Keyboard
from adafruit_hid.keyboard_layout_us import KeyboardLayoutUS
from adafruit_hid.keycode import Keycode

import neopixel

#  setup for LED to indicate BLE connection
blue_led = digitalio.DigitalInOut(board.BLUE_LED)
blue_led.direction = digitalio.Direction.OUTPUT


# Define the one NeoPixel on nRF52
pixels = neopixel.NeoPixel(board.NEOPIXEL, 1, brightness=0.3, auto_write=True)
color_a = 255
color_b = 0
color_c = 0


STX=0x02
ETX=0x03

uart = busio.UART(board.TX, board.RX, baudrate=115200)

hid = HIDService()

device_info = DeviceInfoService(software_revision=adafruit_ble.__version__,
                                manufacturer="Adafruit Industries")
advertisement = ProvideServicesAdvertisement(hid)
advertisement.appearance = 961
scan_response = Advertisement()
scan_response.complete_name = "CircuitPython HID"

ble = adafruit_ble.BLERadio()

if not ble.connected:
    print("advertising")
    ble.start_advertising(advertisement, scan_response)
else:
    print("already connected")
    print(ble.connections)

keyboard = Keyboard(hid.devices)
keyboard_layout = KeyboardLayoutUS(keyboard)

while True:
    while not ble.connected:
        time.sleep(0.2)
        blue_led.value = not blue_led.value
        pixels.fill((0, 0, 0))
    print("Start typing:")

    while ble.connected:
        blue_led.value = True
        data = uart.read(11)
        if data is not None:
            if (len(data) == 11) and (data[0] == STX) and (data[1] == 0x08) and (data[10] == ETX):
                keyboard.report=bytearray(data[2:10])
                keyboard.send()
                pixels.fill((color_a, color_b, color_c))
                color_a, color_b, color_c = (color_c, color_a, color_b)
                print(data[2:10])
            elif len(data)>0:
                # Scan for STX ... ETX to resync
                print(data)
                report = bytearray(11)
                for i in range(0, len(data)):
                    if data[i] == STX:
                        report = data[i:len(data)] + uart.read(11-(len(data)-i))
                        print(report)
                        if (len(report) == 11) and (report[0] == STX) and (report[1] == 0x08) and (report[10] == ETX):
                            keyboard.report=bytearray(report[2:10])
                            keyboard.send()

    ble.start_advertising(advertisement)
