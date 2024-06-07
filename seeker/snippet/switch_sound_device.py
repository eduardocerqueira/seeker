#date: 2024-06-07T16:48:29Z
#url: https://api.github.com/gists/bcf8df26ec2c420174fd8d12203ef6f3
#owner: https://api.github.com/users/FLYyyyy2016

import subprocess
import re
import os
import sys
import pyaudio
exe_path=r"C:\Users\liufe\scoop\apps\nircmd\current\nircmd.exe"

def get_current_device():
    # Get the current default audio device
    p = pyaudio.PyAudio()
    info = p.get_default_output_device_info()
    return info['name']


def set_device(device_name):
    subprocess.run([exe_path, 'setdefaultsounddevice', device_name])

def main():
    device1 = "pc"
    device2 = "head"

    current_device = get_current_device()

    if re.search(device1, current_device, re.IGNORECASE):
        set_device(device2)
        print(f"Switched to {device2}")
    else:
        set_device(device1)
        print(f"Switched to {device1}")
if __name__ == "__main__":
    main()