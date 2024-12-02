#date: 2024-12-02T16:51:34Z
#url: https://api.github.com/gists/81eef2bab4bb130bd2034d95f9c22b1d
#owner: https://api.github.com/users/jepler

# SPDX-FileCopyrightText: Copyright (c) 2024 Jeff Epler for Adafruit Industries LLC
#
# SPDX-License-Identifier: MIT

"""Work around the RP2350 E9 erratum by turning off input buffer

A silicon bug in the RP2350 "A2" stepping (the latest and only available version in
December 2024) prevents input pull-downs larger than about 8.2kOhm from working
properly, including the built-in pull down resistors.

A workaround is proposed (keep the input buffer disabled except when actually
reading the input pin value). For various reasons, CircuitPython chose not to
implement this workaround in the core. This workaround slows access to digital
pins and can't work with peripherals like SPI and PIO.

However, in limited circumstances, it is useful to be able to slowly read a digital
pin with a weak pull-down that is affected by the "E9 Erratum" (search for
"RP2350-E9" in https://datasheets.raspberrypi.com/rp2350/rp2350-datasheet.pdf for more
details)

This class implements the workaround in pure Python, using the memorymap module to
directly access the "input enable" bit in the pad control registers.

Typical usage, Feather RP2350 with nothing connected to D5:

.. code-block:: python
    >>> import digitalioe9
    >>> import digitalio
    >>> import board
    >>> 
    >>> d = digitalioe9.DigitalInOutE9(board.D5)
    >>> d.switch_to_input(digitalio.Pull.UP)
    >>> print(d.value)
    True
    >>> d.switch_to_input(digitalio.Pull.DOWN)
    >>> print(d.value)
    False

"""

import os
import struct
import microcontroller.pin
import digitalio
import memorymap

if not os.uname().nodename.startswith("rp2350"):
    raise RuntimeError(
        "This module is only compatible with rp2350, not {os.uname().nodename}"
    )

_RP2350_PADS_BANK0_BASE = const(0x40038000)
_ATOMIC_SET_OFFSET = const(0x2000)
_ATOMIC_CLEAR_OFFSET = const(0x3000)

_input_enable_bit = struct.pack("I", 1 << 6)


def _pin_number(p):
    for i in range(48):
        a = getattr(microcontroller.pin, f"GPIO{i}", None)
        if p is a:
            return i
    raise ValueError("{p!r} is not a GPIO pin")


def _get_reg(offset):
    return memorymap.AddressRange(start=_RP2350_PADS_BANK0_BASE + offset, length=4)


class DigitalInOutE9:
    """A class that functions similar to DigitalInOut, but implements the E9 workaround

    See `digitalio.DigitalInOut` for documentation.
    """
    def __init__(self, pin):
        self._dio = digitalio.DigitalInOut(pin)
        pin_number = _pin_number(pin)
        ctrl_offset = 4 + 4 * pin_number
        self._set_reg = _get_reg(_ATOMIC_SET_OFFSET | ctrl_offset)
        self._clear_reg = _get_reg(_ATOMIC_CLEAR_OFFSET | ctrl_offset)
        self._disable_buffer()

    def _enable_buffer(self):
        self._set_reg[:] = _input_enable_bit

    def _disable_buffer(self):
        self._clear_reg[:] = _input_enable_bit

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.deinit()

    def deinit(self):
        self._dio.deinit()

    def switch_to_input(self, pull=None):
        self._dio.switch_to_input(pull)
        self._disable_buffer()

    def switch_to_output(self, value=False, drive_mode=digitalio.DriveMode.PUSH_PULL):
        self._dio.switch_to_output(value, drive_mode)
        self._disable_buffer()

    @property
    def direction(self):
        return self._dio.direction

    @direction.setter
    def direction(self, value):
        self._dio.direction = direction
        self._disable_buffer()

    @property
    def pull(self):
        return self._dio.pull

    @pull.setter
    def pull(self, value):
        self._dio.pull = pull
        self._disable_buffer()

    @property
    def value(self):
        self._dio.value  # side effect: checks for deinit
        self._enable_buffer()
        result = self._dio.value
        self._disable_buffer()
        return result

    @value.setter
    def value(self, value):
        self._dio.value = value
        self._disable_buffer()
