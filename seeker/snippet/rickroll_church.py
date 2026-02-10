#date: 2026-02-10T17:48:14Z
#url: https://api.github.com/gists/fc40f9860b1b763dce84bc43975a418f
#owner: https://api.github.com/users/bendymind

import time
import webbrowser
from datetime import datetime

# --- SETTINGS ---
CHURCH_TIME = "07:30"
RICK_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ?autoplay=1"

print("System Monitoring... Rickroll standby active for Sunday mornings.")

while True:
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    day_of_week = now.weekday() # Monday is 0, Sunday is 6

    # Only trigger if it's Sunday (6) AND the specific time
    if day_of_week == 6 and current_time == CHURCH_TIME:
        print("IT'S SUNDAY. RICK IS WAITING.")
        webbrowser.open(RICK_URL)
        
        # Sleep for 24 hours so it doesn't trigger again until next week
        time.sleep(86400) 
    
    # Check every 30 seconds
    time.sleep(30)