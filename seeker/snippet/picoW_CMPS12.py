#date: 2023-03-03T16:50:10Z
#url: https://api.github.com/gists/441b1e9e4e7ec95412c0e1061464ee13
#owner: https://api.github.com/users/pwesson

# Raspberry Pi Pico W - CMPS12 digital compass
# Copyright (C) 2023 https://www.roboticboat.uk
# e156e28e-ce3e-49e8-9e22-5b7a61f62125
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# These Terms shall be governed and construed in accordance with the laws of 
# England and Wales, without regard to its conflict of law provisions.
#
# User Interface https://thonny.org/
#

# Register Function
# 0        Command register (write) / Software version (read)

# 1        Compass Bearing as a byte, i.e. 0-255 for a full circle
# 2,3      Compass Bearing as a word, i.e. 0-3599 for a full circle, representing 0-359.9 degrees. Register 2 being the high byte

# 4        Pitch angle - signed byte giving angle in degrees from the horizontal plane, Kalman filtered with Gyro
# 5        Roll angle - signed byte giving angle in degrees from the horizontal plane, Kalman filtered with Gyro

# 6,7      Magnetometer X axis raw output, 16 bit signed integer with register 6 being the upper 8 bits
# 8,9      Magnetometer Y axis raw output, 16 bit signed integer with register 8 being the upper 8 bits
# 10,11    Magnetometer Z axis raw output, 16 bit signed integer with register 10 being the upper 8 bits

# 12,13    Accelerometer  X axis raw output, 16 bit signed integer with register 12 being the upper 8 bits
# 14,15    Accelerometer  Y axis raw output, 16 bit signed integer with register 14 being the upper 8 bits
# 16,17    Accelerometer  Z axis raw output, 16 bit signed integer with register 16 being the upper 8 bits

# 18,19    Gyro X axis raw output, 16 bit signed integer with register 18 being the upper 8 bits
# 20,21    Gyro Y axis raw output, 16 bit signed integer with register 20 being the upper 8 bits
# 22,23    Gyro Z axis raw output, 16 bit signed integer with register 22 being the upper 8 bits

import machine
import time
from micropython import const

# CMPS12 compass registers
i2cAddress = const(0x60)

CONTROL_REGISTER = const(0)

BEARING_REGISTER = const(2) 
PITCH_REGISTER = const(4)
ROLL_REGISTER = const(5)

MAGNET_X_REGISTER = const(6)
MAGNET_Y_REGISTER = const(8)
MAGNET_Z_REGISTER = const(10)

ACCEL_X_REGISTER = const(12)
ACCEL_Y_REGISTER = const(14)
ACCEL_Z_REGISTER = const(16)

GYRO_X_REGISTER = const(18)
GYRO_Y_REGISTER = const(20)
GYRO_Z_REGISTER = const(22)


# i2c data
i2cdata = machine.Pin(0)

# i2c clock
i2cclock = machine.Pin(1)

# Start the i2c network
i2c = machine.I2C(0, sda=i2cdata, scl=i2cclock, freq=100000)

# Function for reading bytes from i2c module registers
def i2cRead(registerAddress, numBytes=1):
     
         # b'\rf' would indicate an error
         return i2c.readfrom_mem(i2cAddress, registerAddress, numBytes)

# CTRL + C to stop the program running.
# Without the try: except it is very hard to stop the infinite loop
try:
    
    # Infinite loop, like on the Arduino
    while (True):
        
        # Capture any i2c network errors
        try:
            
            # Receive 4 bytes
            # We start at the BEARING_REGISTER but each adjacent byte
            # will return the contents of the PITCH_REGISTER and ROLL_REGISTER
            receivedBytes = i2cRead(BEARING_REGISTER, 4)
    
        except Exception as e:
            # Update the User
            # b'\r\xc0' would be an error
            print(e)
            print(hex(receivedBytes[0]))
            print(hex(receivedBytes[1]))       
            print(hex(receivedBytes[2]))       
            print(hex(receivedBytes[3]))       
            continue
        
        # Calculate full bearing (float)
        # Shift the byteHigh bits to the left. Multiply by 2^8
        bearing = ((receivedBytes[0]<<8) + receivedBytes[1]) / 10;
        
        # Compass pitch - byte [0 to 255]
        pitch = receivedBytes[2]
        
        # Need to convert unsigned byte into signed
        if pitch > 127:
            pitch -= 256
        
        # Compass roll - byte [0 to 255]
        roll = receivedBytes[3]
        
        # Need to convert unsigned byte into signed
        if roll > 127:
            roll -= 256
    
        # Update the User
        print("Bearing",bearing, "Pitch", pitch, "Roll", roll)
        
        # Need some delay else the print overruns Thonny screen updates
        time.sleep(0.1);
        
        
# Interrupt handing
# Capture the CTRL + C event
except (KeyboardInterrupt, SystemExit):
    
    # Update the user
    print ("Well done. Finished\n");
