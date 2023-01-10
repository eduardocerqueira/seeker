#date: 2023-01-10T17:01:53Z
#url: https://api.github.com/gists/01864a123bdb437e26278d98229f2860
#owner: https://api.github.com/users/jabastien

"""
Demo:
A USB keyboard is connected (via USB OTG cable) to a Trinked M0.
Trinked run KBDADVUARTUSBH: Keyboard Advanced UART USB Host:
 https://github.com/gdsports/usbhostcopro/tree/master/KBDADVUARTUSBH
The Trinked send it via UART TX to the UART RX of a nRF52840.
Every keyboard HID code is transmitted over USB and BLE simultaneously.
"""
import time
import board
from digitalio import DigitalInOut, Direction
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

usb_keyboard = Keyboard(usb_hid.devices)
usb_keyboard_layout = KeyboardLayoutUS(usb_keyboard)

while True:
    while not ble.connected:
        pass
    print("Start typing:")

    while ble.connected:

        data = uart.read(11)
        if data is not None:
            if (len(data) == 11) and (data[0] == STX) and (data[1] == 0x08) and (data[10] == ETX):
                keyboard.report=bytearray(data[2:10])
                keyboard.send()
                usb_keyboard.report=bytearray(data[2:10])
                usb_keyboard.send()
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
                            usb_keyboard.report=bytearray(report[2:10])
                            usb_keyboard.send()

    ble.start_advertising(advertisement)
