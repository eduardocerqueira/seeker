#date: 2026-02-10T17:48:14Z
#url: https://api.github.com/gists/fc40f9860b1b763dce84bc43975a418f
#owner: https://api.github.com/users/bendymind

import time
import webbrowser
import random

# --- SETTINGS ---
ALARM_TIME = "07:30"
CHURCH_URL = "https://www.google.com/maps" # Or your church's website
ALARM_SOUNDS = [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ", # Classic Rickroll
    "https://www.youtube.com/watch?v=3Wn0Xp4N_No", # High-energy music
]
REMINDER_TEXT = "GET OUT OF BED AND GO TO CHURCH!"

print(f"Monitoring... The 'Church-Prompt' is set for {ALARM_TIME}.")

def trigger_alarm(iteration):
    print(f"\n{REMINDER_TEXT}")
    print(f"This is alert number {iteration}!")
    
    # Open a random loud video
    webbrowser.open(random.choice(ALARM_SOUNDS))
    
    # On the 3rd 'nag', open the map/website to make the goal visual
    if iteration >= 3:
        print("Opening your destination for extra motivation...")
        webbrowser.open(CHURCH_URL)

# --- MAIN LOOP ---
while True:
    current_time = time.strftime("%H:%M")
    
    if current_time == ALARM_TIME:
        nag_count = 1
        # This nested loop 'nags' you every 3 minutes until you kill the script
        while True:
            trigger_alarm(nag_count)
            print("Snooze is NOT an option. To stop this, close the terminal/PyCharm.")
            
            time.sleep(180) # Wait 3 minutes before the next nag
            nag_count += 1
            
    time.sleep(30) # Check the clock every 30 seconds