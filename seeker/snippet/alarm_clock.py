#date: 2026-02-10T17:48:14Z
#url: https://api.github.com/gists/fc40f9860b1b763dce84bc43975a418f
#owner: https://api.github.com/users/bendymind

import time
import webbrowser

# Set your desired alarm time (24-hour format)
ALARM_TIME = "07:30" 
# A loud or energetic link (like a "Get Out of Bed" playlist)
ALARM_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ" 

print(f"Alarm set for {ALARM_TIME}. Waiting...")

while True:
    current_time = time.strftime("%H:%M")
    
    if current_time == ALARM_TIME:
        print("WAKE UP! (Close the script to stop the loop)")
        # This opens your default browser to the URL
        webbrowser.open(ALARM_URL)
        
        # Wait 60 seconds so it doesn't open 100 tabs in one minute
        time.sleep(60) 
        
        # Optional: Add a second wait or a 'nag' feature
        print("I'll check back in 5 minutes to see if you're up...")
        time.sleep(300) 
    
    # Check the time every 10 seconds to save CPU
    time.sleep(10)