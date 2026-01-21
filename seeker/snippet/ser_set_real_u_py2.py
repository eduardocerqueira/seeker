#date: 2026-01-21T17:45:08Z
#url: https://api.github.com/gists/646e24353ffc39b1a299ea69286a401f
#owner: https://api.github.com/users/dipcore

from __future__ import print_function

import serial
from serial import PARITY_NONE, STOPBITS_ONE, EIGHTBITS
import serial.tools.list_ports, time, datetime, signal

version = "1.02.00"
G_IS_EXIT = False
bps = 2000000


def handle_signal(signum, frame):
    global G_IS_EXIT
    try:
        G_IS_EXIT = True
    except Exception as e:
        try:
            print("error4: %s" % e)
        finally:
            e = None
            del e


def show_real_u():
    """
    Loop through the port list and send a screen-on command to each port.
    If it returns OK the port is usable; continue sending commands to display the real USB drive.
    Otherwise, skip.
    """

    print("Function: display real disks bps:%d Ver:%s \n" % (bps, version))

    while True:
        print(
            "Scanning port %s"
            % datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        )

        port_list = list(serial.tools.list_ports.comports())
        for port_obj in port_list:
            try:
                ser = serial.Serial(
                    port_obj[0],
                    bps,
                    timeout=0.5,
                    stopbits=STOPBITS_ONE,
                    bytesize=EIGHTBITS,
                    parity=PARITY_NONE,
                    write_timeout=1,
                )
            except Exception as e:
                try:
                    pass
                finally:
                    e = None
                    del e
                continue

            time.sleep(1)
            resp = ser.readline()
            time.sleep(0.5)
            resp = ser.readline()
            time.sleep(0.5)

            if not ser.is_open:
                continue

            try:
                try:
                    ser.write(b"AT+WRX=always\x00d\x00a")
                    resp = ser.readline()
                    if resp != b"OK\r\n":
                        ser.close()
                        continue

                    time.sleep(0.5)


                    print("[%s]: Connection completed..." % port_obj[0])

                    res = ser.write(b"AT+WRX=realUpan\x00d\x00a")
                    resp = ser.readline()
                    time.sleep(0.5)

                    res = ser.write(b"AT+WRX=mass\x00d\x00a")
                    time.sleep(0.5)

                    print("[%s]: Success..." % port_obj[0])

                    time.sleep(3)
                except Exception as e:
                    try:
                        pass
                    finally:
                        e = None
                        del e
            finally:
                ser.close()

        time.sleep(1)


if __name__ == "__main__":
    show_real_u()
