#date: 2023-03-01T16:44:11Z
#url: https://api.github.com/gists/459934ec9b3c448f30b1d6264f2bb2b0
#owner: https://api.github.com/users/demodude4u

__version__ = "0.0.1"
__author__ = "Demod"

import random
import time
from machine import Pin, UART

try:
    from thumbyGrayscale import display
    grayscale = True
except ImportError:
    from thumby import display
    grayscale = False

try:
    import emulator
    emulated = True
except ImportError:
    emulated = False


class Link:
    mode = 0  # 0 = write first, 1 = read first
    syncFrames = 0

    def __init__(self):
        self.uart = UART(0, baudrate=115200, rx=Pin(1, Pin.IN), tx=Pin(
            0, Pin.OUT), timeout=1000, txbuf=1, rxbuf=1)
        Pin(2, Pin.OUT).value(1)
        while (self.uart.any() > 0):
            self.uart.read(1)

    def tryHandshake(self):
        self.uart.write(bytearray([0x80]))
        self.uart.read(1)  # echo
        time.sleep(0.1)  # enough time for a response
        while (self.uart.any() > 0):
            response = self.uart.read(1)[0]
            if (response == 0x81):  # HandshakeAck
                self.mode = 1
                return True
        return False

    def tryHandshakeAck(self):
        while (self.uart.any() > 0):
            response = self.uart.read(1)[0]
            if (response == 0x80):  # Handshake
                self.uart.write(bytearray([0x81]))
                self.uart.read(1)  # echo
                self.mode = 0
                return True
        return False

    def sync(self, data):
        self.syncFrames += 1
        self.waitCount = 0
        if (self.mode == 0):  # write first
            self.uart.write(bytearray([data]))
            self.uart.read(1)  # echo

            while (self.uart.any() == 0):
                self.waitCount += 1

            return self.uart.read(1)[0]

        else:  # read first
            while (self.uart.any() == 0):
                self.waitCount += 1
            ret = self.uart.read(1)

            self.uart.write(bytearray([data]))
            self.uart.read(1)  # echo

            return ret[0]

    def clear(self):
        while (self.uart.any() > 0):
            self.uart.read(1)


class EmuLink:
    syncFrames = 0

    def tryHandshake(self):
        self.mode = 0
        return random.random() < 0.3

    def tryHandshakeAck(self):
        self.mode = 1
        return random.random() < 0.01

    def sync(self, data):
        self.syncFrames += 1
        return data

    def clear(self):
        pass


def linkConnectUI():
    if (emulated):
        link = EmuLink()
    else:
        link = Link()

    display.setFont("/lib/font3x5.bin", 3, 5, 2)
    display.drawText("CONNECTING", 36 - 23, 10, 1)
    HandshakeWait = 0
    while (True):
        if (HandshakeWait >= 30):
            HandshakeWait = 0
            if (link.tryHandshake()):
                break
        else:
            if (link.tryHandshakeAck()):
                break

        HandshakeWait += 1

        lineX = 36 + HandshakeWait
        display.drawLine(lineX-1, 20, lineX-1, 30, 0)
        display.drawLine(72-lineX+1, 20, 72-lineX+1, 30, 0)
        display.drawLine(lineX, 20, lineX, 30, 1)
        display.drawLine(72-lineX, 20, 72-lineX, 30, 1)
        display.update()

    display.drawText("PLAYER "+str(link.mode+1), 36 - 19, 35, 1)
    display.update()
    link.clear()
    time.sleep(1)

    return link
