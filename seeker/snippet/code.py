#date: 2023-02-28T16:57:06Z
#url: https://api.github.com/gists/8da2f7ab98c422554cb221975dd4a23c
#owner: https://api.github.com/users/jepler

import time
import os
import struct

import gifio
import board
import busio
import displayio
import storage

TEST_NUMBER = 3

print("Initializing display")
displayio.release_displays()
spi = busio.SPI(MOSI=board.LCD_MOSI, clock=board.LCD_CLK)
display_bus = displayio.FourWire(
    spi,
    command=board.LCD_D_C,
    chip_select=board.LCD_CS,
    reset=board.LCD_RST,
    baudrate=80_000_000,
)
_INIT_SEQUENCE = (
    b"\x01\x80\x80"  # Software reset then delay 0x80 (128ms)
    b"\xEF\x03\x03\x80\x02"
    b"\xCF\x03\x00\xC1\x30"
    b"\xED\x04\x64\x03\x12\x81"
    b"\xE8\x03\x85\x00\x78"
    b"\xCB\x05\x39\x2C\x00\x34\x02"
    b"\xF7\x01\x20"
    b"\xEA\x02\x00\x00"
    b"\xc0\x01\x23"  # Power control VRH[5:0]
    b"\xc1\x01\x10"  # Power control SAP[2:0];BT[3:0]
    b"\xc5\x02\x3e\x28"  # VCM control
    b"\xc7\x01\x86"  # VCM control2
    b"\x36\x01\x40"  # Memory Access Control
    b"\x37\x01\x00"  # Vertical scroll zero
    b"\x3a\x01\x55"  # COLMOD: Pixel Format Set
    b"\xb1\x02\x00\x18"  # Frame Rate Control (In Normal Mode/Full Colors)
    b"\xb6\x03\x08\x82\x27"  # Display Function Control
    b"\xF2\x01\x00"  # 3Gamma Function Disable
    b"\x26\x01\x01"  # Gamma curve selected
    b"\xe0\x0f\x0F\x31\x2B\x0C\x0E\x08\x4E\xF1\x37\x07\x10\x03\x0E\x09\x00"  # Set Gamma
    b"\xe1\x0f\x00\x0E\x14\x03\x11\x07\x31\xC1\x48\x08\x0F\x0C\x31\x36\x0F"  # Set Gamma
    b"\x11\x80\x78"  # Exit Sleep then delay 0x78 (120ms)
    b"\x29\x80\x78"  # Display on then delay 0x78 (120ms)
)

display = displayio.Display(display_bus, _INIT_SEQUENCE, width=320, height=240)

d = gifio.OnDiskGif('demo.gif')

if TEST_NUMBER == 1:
# Test 1:
    d = gifio.OnDiskGif('demo.gif')
    t0 = time.monotonic_ns()
    for _ in range(100):
        d.next_frame()
    t1 = time.monotonic_ns()
    print(f"Time to LOAD 100 frames = {(t1-t0) / 1e9}s")

# Test 2:
elif TEST_NUMBER == 2:
    d = gifio.OnDiskGif('demo.gif')
    ow = (display.width - d.width) // 2
    oh = (display.height - d.height) // 2
    t0 = time.monotonic_ns()
    for _ in range(100):
        display_bus.send(42, struct.pack(">hh", ow, d.width + ow - 1))
        display_bus.send(43, struct.pack(">hh", oh, d.height + ow - 1))
        display_bus.send(44, d.bitmap)
        d.next_frame()

    t1 = time.monotonic_ns()
    print(f"Time to DirectIO 100 frames = {(t1-t0) / 1e9}s")

# Test 2:
elif TEST_NUMBER == 3:
    d = gifio.OnDiskGif('demo.gif')
    splash = displayio.Group()
    display.root_group = splash


    face = displayio.TileGrid(d.bitmap,
                              pixel_shader=displayio.ColorConverter
                              (input_colorspace=displayio.Colorspace.RGB565_SWAPPED))
    splash.append(face)

    ow = (display.width - d.width) // 2
    oh = (display.height - d.height) // 2
    t0 = time.monotonic_ns()
    for _ in range(100):
        display_bus.send(42, struct.pack(">hh", ow, d.width + ow - 1))
        display_bus.send(43, struct.pack(">hh", oh, d.height + ow - 1))
        display_bus.send(44, d.bitmap)
        d.next_frame()
        display.refresh()

    t1 = time.monotonic_ns()
    print(f"Time to DisplayIO 100 frames = {(t1-t0) / 1e9}s")

