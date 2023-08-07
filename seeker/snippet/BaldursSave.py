#date: 2023-08-07T16:57:01Z
#url: https://api.github.com/gists/b3e6c244aae8f31dbf02ba00221a3641
#owner: https://api.github.com/users/TylerCode

import psutil
from pynput.keyboard import Key, Controller

def check_if_process_running(process_keywords):
    try:
        # Check every process cause this wasn't working.... idk why
        for proc in psutil.process_iter():
            try:
                # Check if process name contains any of the keywords.
                if any(keyword.lower() in proc.name().lower() for keyword in process_keywords):
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return False
    except Exception as e:
        print(e)
        return False

process_keywords = ["bg3", "baldur"]
keyboard = Controller()

if check_if_process_running(process_keywords):
    # Press and release F5 to fucking save because GD Dave didn't think autosave was important
    keyboard.press(Key.f5)
    keyboard.release(Key.f5)
    print(f"A process with keywords {process_keywords} is running, pressed F5.")
else:
    print(f"No process with keywords {process_keywords} is running.")
