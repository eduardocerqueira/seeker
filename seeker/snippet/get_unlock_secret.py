#date: 2021-10-06T16:53:17Z
#url: https://api.github.com/gists/cc2e151d48bdf983672deafa7f37c56f
#owner: https://api.github.com/users/z3ntu

#!/usr/bin/python3

import sys

def get_secret(serial: str) -> str:
    secret = ""
    for i in range(16):
        value = ord(serial[i]) + ord("AB2D3F3B37890C1A"[i]) + 9
        secret += format(value, 'X')
    return secret

# Get serial number - in adb shell:
# $ cat /sys/sys_info/serial_number

# Errors:
# 0x7000 - ERR_UNLOCK_KEY_WRONG_LENGTH
# 0x7001 - ERR_UNLOCK_WRONG_KEY_CODE
# 0x100d - PART_ERASE_FAIL

# Unlock failed - Err:0x7000


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} SERIAL_NUMbER")
        sys.exit(1)
    if len(sys.argv[1]) != 16:
        print("ERROR: The given serial number must be 16 characters long!")
        sys.exit(1)
    print("Calculated unlock secret:")
    secret = get_secret(sys.argv[1])
    print()
    print("Boot into bootloader mode (Vol+Up during boot + fastboot)")
    print(f"fastboot oem key {secret}")
    print("fastboot oem unlock")
