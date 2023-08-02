#date: 2023-08-02T16:58:15Z
#url: https://api.github.com/gists/80084e998000f5ba1b0305486feafdb2
#owner: https://api.github.com/users/Jpk518

# Check for Marine Atlantic Ferry Availability

# Crontabs
#SHELL=/bin/bash
#*/1 * * * * source /Users/<USER>/<PROJECT_PATH>/venv/bin/activate && python3 /Users/<USER>/<PROJECT_PATH>/venv/bin/ferry-checker.py

import requests
import json
import time
import sys
import pync
from datetime import datetime


requestURLs = [
    "<REQUEST_URL>"
]

def calc_available_departures():
    for url in requestURLs:
        try:
            response = requests.get(url=url)
            departures = json.loads(response.text)["hydra:member"][0]["departures"]

            # Cycle through ferry departures
            for departure in departures:
                departureDate = departure["departureDate"]
                departureTime = departure["departureTime"]
                shipCode = departure["shipCode"]

                # Filter out MV Atlantic Vision
                if shipCode != 'VIS':
                    # Cycle through all car size options for departure
                    resources = departure["resources"]
                    for resource in resources:

                        # Filter out all car options besides regular vehicle
                        if resource["supplierCode"] == "PBNS" and resource["resourceCode"] == "ATL":

                            totalCapacity = float(resource["totalCapacity"])
                            bookedAmount = float(resource["bookedAmount"])
                            freeCapacity = float(resource["freeCapacity"])

                            # Filter out unavailable options based on capacity
                            if(freeCapacity > 0.0 or totalCapacity - bookedAmount > 16.0):

                                old_stdout = sys.stdout
                                log_file = open("ferry-availability.log","a")
                                sys.stdout = log_file

                                current_time = datetime.now().strftime("%H:%M:%S")
                                print(f'{current_time} - Seat available for: {departureDate} - {departureTime}')

                                sys.stdout = old_stdout

                                date = '2023-08-07'
                                if(departureDate == f'{date}T00:00:00'):
                                    pync.notify(f'A new appointment is available for: {departureDate} - {departureTime}', sound='default')

        except Exception as e:
            pync.notify(e, sound='default')

print('Running script')
calc_available_departures()

time.sleep(15)
calc_available_departures()

# time.sleep(15)
# calc_available_departures()

print('Ending script')