#date: 2022-11-10T17:15:50Z
#url: https://api.github.com/gists/853e71803ff74b3044b423c5f23adde2
#owner: https://api.github.com/users/sergiomarchio

from pyautogui import typewrite
import time
import sys

def keepalive(interval=60):
    """
    Keeps alive some apps by sending keystrokes :)
    After executing this script, you mus foucs some input field inside the target app.
    
    Parameters
    ----------
    interval : int
        The time interval in seconds to wait for the next key to be pressed.
    """
    print("keeping ... alive")
    print("Press Ctrl+C to end script")
    i = 0
    while True:
        for s in range(interval):
            print(f"Waiting... {s} of {interval}    ", end ="\r")
            time.sleep(1)
        typewrite(f" {i}")
        i += 1
    
    print()
    print("finished! now it can die... :(")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        keepalive(int(sys.argv[1]))
    else:
        keepalive()
