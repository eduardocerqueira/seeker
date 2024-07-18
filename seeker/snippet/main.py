#date: 2024-07-18T16:56:22Z
#url: https://api.github.com/gists/2583c221cfd22708b77ec1ae5d481c8d
#owner: https://api.github.com/users/Williangalvani

#!/usr/bin/env python3

from mavlink2rest_helper import Mavlink2RestHelper
import time

MAVLINK2REST_ADDRESS = "http://192.168.15.132/mavlink2rest"
helper = Mavlink2RestHelper(MAVLINK2REST_ADDRESS)

def set_servo_params():
    """Set all SERVO_FUNCTIONS to RCIN"""
    for servo in range(1, 17):
        function = servo + 50
        print(f"Setting SERVO{servo}_FUNCTION to {function}")
        helper.set_param(f"SERVO{servo}_FUNCTION", "MAV_PARAM_TYPE_UINT8", function)
        time.sleep(0.1)

def main():
    print("Starting direct_thruster_control")
    print("Setting parameters")
    set_servo_params()

    while True:
        print("Moving up...")
        for _ in range(10):
            helper.send_rc_override([1550] * 16)
            time.sleep(0.2)

        print("Moving down...")
        for _ in range(10):
            helper.send_rc_override([1450] * 16)
            time.sleep(0.2)

        print("Done")

if __name__ == "__main__":
    main()