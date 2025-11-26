#date: 2025-11-26T16:50:56Z
#url: https://api.github.com/gists/8feba96ccb4fb19d96708f35ab92d518
#owner: https://api.github.com/users/rangeldarosa

import time
import subprocess

try:
    print("Press Ctrl+C to stop the script.")
    while True:
        # Use Scroll Lock key - it registers as activity but doesn't produce visible output
        subprocess.run(['powershell.exe', '-WindowStyle', 'Hidden', '-Command',
                       'Add-Type -AssemblyName System.Windows.Forms; ' +
                       '[System.Windows.Forms.SendKeys]::SendWait("{SCROLLLOCK}{SCROLLLOCK}")'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        time.sleep(30)  # Check every 30 seconds
except KeyboardInterrupt:
    print("\nScript stopped by user.")