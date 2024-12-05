#date: 2024-12-05T17:02:41Z
#url: https://api.github.com/gists/276d9d4183df0e6b3431dcc49a6adbcd
#owner: https://api.github.com/users/fl-kmarston

#!/usr/bin/env python3

import sys
import os
import logging
import argparse
import time

from threading import current_thread, Thread, Lock
from twisted.internet import reactor

# pymodbus~=2.5.3
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext
from pymodbus.datastore import ModbusSparseDataBlock
from pymodbus.server.asynchronous import StartTcpServer

from random import randrange

# Function to set logging level for all loggers
def set_logging_level(level):
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if logger.name.startswith("pyModbus"):
            continue
        if logger.name.startswith("pymodbus"):
            continue
        if logger.name.startswith("asyncio"):
            continue
        if logger.name.startswith("concurrent"):
            continue
        logger.setLevel(level)


class SimulateModbusUnit(ModbusSlaveContext):

    def __init__(self, *args, **kwargs):
        '''
            'di' - Discrete Inputs initializer
            'co' - Coils initializer
            'hr' - Holding Register initializer
            'ir' - Input Registers iniatializer

            zero_mode
            '''
        super().__init__(*args, **kwargs)
        self._dataLock = Lock()
        self._IModbusSlaveContext__fx_mapper['di'] = 'd'
        self._IModbusSlaveContext__fx_mapper['co'] = 'c'
        self._IModbusSlaveContext__fx_mapper['hr'] = 'h'
        self._IModbusSlaveContext__fx_mapper['ir'] = 'i'
        self._IModbusSlaveContext__fx_mapper['d'] = 'd'
        self._IModbusSlaveContext__fx_mapper['c'] = 'c'
        self._IModbusSlaveContext__fx_mapper['h'] = 'h'
        self._IModbusSlaveContext__fx_mapper['i'] = 'i'

    def setValues(self, fx, address, values):
        '''Make SetValues Thread safe,
        Add hr,co,di,'''
        print("setValues", fx, address, values)
        with self._dataLock:
            return super().setValues(fx, address, values)

    def loop(self):
        while reactor.running:
            _loop_start = time.time()
            wait = randrange(5) + 1
            # print(f"Delay: {current_thread().name} {wait}")
            time.sleep(wait)
            diff = time.time() - _loop_start -wait
            print(f"{diff:0.3f} Done: {current_thread().name}   Wait: {wait}")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":

    FORMAT = (
        "%(threadName)-11s" " %(levelname)-8s %(module)-12s:%(lineno)-5s %(message)s"
    )
    logging.basicConfig(format=FORMAT)
    level = os.getenv("LOGLEVEL", "WARNING")

    print(f"LOGLEVEL={level}")
    set_logging_level(level)

    log.warning("warning...")
    log.info("info...")
    log.debug("debug...")

    args = parse_args()
    log.info("Passed args: {}".format(args))

    addr = args.local
    host, port = addr.split(":")
    port = int(port)

    hr1 = ModbusSparseDataBlock(values={0: [12] * 20, 4090: [13] * 95})
    unit1 = SimulateModbusUnit(hr=hr1, zero_mode=True)  # Keep it consistent!

    # hr2 = ModbusSparseDataBlock(values={0: [12] * 20, 4090: [13] * 95})
    unit2 = SimulateModbusUnit(zero_mode=True)  # Keep it consistent!

    the_loopers = [
        unit1.loop,
        unit2.loop,
    ]

    def reactor_loop(unit_loops):
        threads = []
        for loop in unit_loops:

            threads.append(
                Thread(
                    target=loop,
                )  # name=f"poll_cycle({cube.name})")
            )
        for thread in threads:
            thread.start()
        print("ALL STARTED...")
        return

    reactor.callInThread(reactor_loop, the_loopers)
    context_server = ModbusServerContext(slaves={1: unit1, 2: unit2}, single=False)
    StartTcpServer(
        context=context_server,
        address=(host, port),
        # framer=MyFramer,
    )
