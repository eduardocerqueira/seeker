#date: 2025-02-11T16:59:04Z
#url: https://api.github.com/gists/acb6c0a0ecfd096eb8d611674aee1c5a
#owner: https://api.github.com/users/damp11113

import board
import busio
import time
import digitalio

IKB_1_ADDR = 0x48
IKB_1_ADDR_2 = 0x49

# Detect board type and set I2C pins accordingly
i2c1 = busio.I2C(board.GP1, board.GP0)
print("waiting iKB-1 (Try reset)")
found = False
while not found:
    while not i2c1.try_lock():
        pass  # Wait until the I2C bus is available

    try:
        devices = i2c1.scan()  # Scan for I2C devices
        if devices:
            for device in devices:
                if device == IKB_1_ADDR or device == IKB_1_ADDR_2:
                    if device == IKB_1_ADDR_2:
                        IKB_1_ADDR = IKB_1_ADDR_2
                        
                    print("Found iKB-1")
                    found = True
    finally:
        i2c1.unlock()  # Release the I2C bus
        
    time.sleep(1)


def write(wd):
    while not i2c1.try_lock():
        pass
    i2c1.writeto(IKB_1_ADDR, bytes(wd))
    i2c1.unlock()

def read(n):
    while not i2c1.try_lock():
        pass
    data = bytearray(n)
    i2c1.readfrom_into(IKB_1_ADDR, data)
    i2c1.unlock()
    return data

def reset(wait=0.1):
    write([0x00])
    time.sleep(wait)

class Pin:
    OUT = digitalio.Direction.OUTPUT
    IN = digitalio.Direction.INPUT
    PULL_UP = digitalio.Pull.UP

    def __init__(self, pin, mode=OUT, pull=None):
        self.pin = max(0, min(7, pin))
        self.pull = pull

    def value(self, d=None):
        if d is None:
            write([0x08 + self.pin, 3 if self.pull == self.PULL_UP else 2])
            return read(1)[0]
        else:
            write([0x08 + self.pin, 1 if d else 0])
            return d

class ADC:
    def __init__(self, pin):
        self.pin = pin
    
    def read(self):
        write([0x80 + (self.pin << 4)])
        time.sleep(0.1)
        d = read(2)
        return (d[0] << 8) | d[1]

class Motor:
    FORWARD = 1
    BACKWARD = 2
    STOP = 3

    def __init__(self, m, direction=STOP, speed=0):
        self._m = max(1, min(4, m))
        self._dir = direction
        self._speed = speed
        self.update()

    def update(self):
        if self._dir == self.FORWARD:
            d = self._speed
        elif self._dir == self.BACKWARD:
            d = ((~self._speed) & 0xFF) + 1
        else:
            d = 0
        write([0x20 | (1 << (self._m - 1)), d])

    def dir(self, direction=None):
        if direction is not None:
            self._dir = direction
        return self._dir

    def speed(self, speed=None):
        if speed is not None:
            self._speed = max(0, min(100, int(speed)))
        return self._speed

class Servo:
    def __init__(self, ch, angle=0):
        self.ch = max(1, min(6, ch))
        self._angle = angle
    
    def angle(self, angle=None):
        if angle is not None:
            self._angle = max(0, min(200, int(angle)))
            write([0x40 | (1 << (self.ch - 1)), self._angle])
        return self._angle

class UART:
    def __init__(self, baudrate=9600):
        self.baudToBit = {2400: 0, 9600: 1, 57600: 2, 115200: 3}.get(baudrate, 1)

    def available(self):
        write([0x01])
        return read(1)[0]
    
    def read(self, nbytes=1):
        nbytes = min(0xFF, max(1, nbytes))
        write([0x02, nbytes])
        return read(nbytes)

    def write(self, buf):
        write([0x04 | self.baudToBit] + list(buf))

# reset()
