#date: 2023-05-23T16:49:37Z
#url: https://api.github.com/gists/c2e1dcd99c12ef8ed20ede5bba48ea28
#owner: https://api.github.com/users/imandrec

import time
import board
from analogio import AnalogIn

analog_in = AnalogIn(board.A1)


while True:
    voltage = analog_in.value *3.3 / 65536
    uvIndex = voltage/0.1
    print("UV Voltage is : " + str(voltage))
    print("uvIndex is : " +str(uvIndex))
    time.sleep(0.1)
