#date: 2021-12-27T16:47:26Z
#url: https://api.github.com/gists/8c0697b1b57852767a90e51e1933f15a
#owner: https://api.github.com/users/superbirds2008

#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#
# Sending a SMS with Python and pyserial via an USB 3G Modem on pfSense/FreeBSD
# tested with: Huawei Technologies Co., Ltd. E620 USB Modem
# and a Telekom SIM card

import serial
from curses import ascii
import time
import sys
import getopt

baudrate = 9600
device = "/dev/cuaU0.0"
simpin = 1234


def getline(ser):
    buf = ""
    while True:
        ch = ser.read(1)
        if ch == '\r':
            break
        buf += ch
    return buf


def send_sms(message, phonenumber):
    # Initialize serial connection to 3G USB Modem
    modem = serial.Serial(device, baudrate, timeout=5)
    print "Connected to " + modem.name
    # Check modem status
    modem.write(b'AT\r')
    sent_cmd = getline(modem)
    response = modem.read(4)
    if "OK" in response:
        print "Modem Status: OK"
        # check pin and enter it if needed
        modem.write(b'AT+CPIN?\r')
        sent_cmd = getline(modem)
        line2 = getline(modem)  # empty
        line3 = getline(modem)  # empty
        response = getline(modem)  # get OK
        if "SIM PIN" in response:
            print "Sending PIN to modem ... "
            modem.write(b'AT+CPIN="%s"' % simpin)
            sent_cmd = getline(modem)
            response = getline(modem)
            print response
        elif "READY" in response:
            print "PIN already entered."
        # set modem to text mode
        modem.write(b'AT+CMGF=1\r')
        sent_cmd = getline(modem)
        response = getline(modem)
        if "OK" in response:
            print "Modem set to text mode!"
            # send sms
            print "Sending sms ..."
            modem.write(b'AT+CMGS="%s"\r' % phonenumber)
            time.sleep(0.5)
            modem.write(b'%s\r' % message)
            time.sleep(0.5)
            modem.write(ascii.ctrl('z'))
            time.sleep(0.5)
            response = modem.readlines()
            if "+CMGS" in "".join(response):
                print "Sucess: SMS sent!"
            else:
                print "Error: SMS not sent!"
        else:
            print "Error: Setting modem to text mode failed!"
    elif "NO CARRIER" in response:
        print "Error: No 3G connection!"
    else:
        print "Error: Something else failed!"


def main(argv):
    message = ''
    phonenumber = ''
    usage = 'usage: send_sms.py -p +123467890 -m "your message"'
    try:
        opts, args = getopt.getopt(argv, "hp:m:", ["phonenumber=", "message="])
    except getopt.GetoptError:
        print usage
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print usage
            sys.exit()
        elif opt in ("-p", "--phonenumer"):
            phonenumber = arg
        elif opt in ("-m", "--message"):
            message = arg
    send_sms(message=message, phonenumber=phonenumber)

if __name__ == '__main__':
    main(sys.argv[1:])
