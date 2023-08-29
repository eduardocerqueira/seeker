#date: 2023-08-29T16:49:41Z
#url: https://api.github.com/gists/f9e78e23d69a3e2f80f6dfcc600ce068
#owner: https://api.github.com/users/honnet

#!/usr/bin/python
import sys
import glob
import serial
import math


DEBUG_PRINT = 0

data_backup = [-1, 1,1,1, 1,1,1]

################################# functions #######################################
def serial_init():
    global DEBUG_PRINT
    devices = glob.glob("/dev/ttyACM*")
    if DEBUG_PRINT:
        print(devices)
    ser = serial.Serial(devices[0], 115200)
    success = ser.isOpen()
    if not success:
        print("\n!!! Error: serial device not found !!!")
        sys.exit(-1)
    return ser

def get_data(ser):
    global DEBUG_PRINT
    global data_backup
    received = ser.readline()
    ser.flush()
    data = received.split()

    if DEBUG_PRINT:
        print("\ndata =", data)

    if len(data) != 7:
        return data_backup
    else:
        data_backup = data
        return data

#################################### main #########################################
def main():
    ser = serial_init()
    # format: "touch mag_x mag_y mag_z acc_x acc_y acc_z"

    while True: # wait for a ctrl-c interrupt
        print("-------------------")
        data = get_data(ser)

        # touch sensing (binary):
        print("touch:", data[0])

        # magnetometer (only needs y and x for now):
        heading = math.atan2(float(data[2]), float(data[1])) # (y, x)
        heading = int(math.degrees(heading))
        print("heading:", heading, "\t", int((180+heading)/5) * "*")

        # accelerometer (only needs x and z for now):
        finger = math.atan2(float(data[4]), float(data[6]))
        finger = int(math.degrees(finger))
        print("finger:", finger, "\t", int((180+finger)/5) * "*")

        print()

if __name__ == "__main__":
    main()
