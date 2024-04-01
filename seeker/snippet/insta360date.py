#date: 2024-04-01T16:50:40Z
#url: https://api.github.com/gists/9dd157a3ce4092e41daae655618e84d9
#owner: https://api.github.com/users/alessandrotedd

import os
import datetime

def change_file_timestamps():
    directory = os.getcwd()
    for filename in os.listdir(directory):
        if filename.startswith(("IMG_", "VID_")):
            try:
                year = int(filename[4:8])
                month = int(filename[8:10])
                day = int(filename[10:12])
                hour = int(filename[13:15])
                minute = int(filename[15:17])
                second = int(filename[17:19])
                
                timestamp = datetime.datetime(year, month, day, hour, minute, second)
                filepath = os.path.join(directory, filename)
                os.utime(filepath, (timestamp.timestamp(), timestamp.timestamp()))
                
                print(f"Timestamp changed for file: {filename}")
            except Exception as e:
                print(f"Error changing timestamp for file {filename}: {e}")
    print("done")

change_file_timestamps()
