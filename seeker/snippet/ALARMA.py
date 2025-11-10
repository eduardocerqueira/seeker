#date: 2025-11-10T17:14:28Z
#url: https://api.github.com/gists/cc948afa077377f57637a64515b78ebc
#owner: https://api.github.com/users/gvbmafia

import psutil
import pyttsx3

utilizare_cpu = psutil.cpu_percent(interval=1)
print(f"Utilizarea CPU: {utilizare_cpu}%")
engine = pyttsx3.init()
if utilizare_cpu > 8.0:
 engine.say(f"Your CPU usage is at {utilizare_cpu} precent")
 engine.runAndWait()
get_property_engine = engine.getProperty('rate')
engine.setProperty('rate', get_property_engine + 100)
