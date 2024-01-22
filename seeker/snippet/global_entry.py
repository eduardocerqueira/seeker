#date: 2024-01-22T16:52:54Z
#url: https://api.github.com/gists/5e9f9a05f83399a3474f0d781920c2e2
#owner: https://api.github.com/users/socks415

#!/usr/bin/env python

import requests
import time
import sys
from datetime import datetime, timedelta
from twilio.rest import Client


# Idea and details located here. I just added SMS capability
# https://packetlife.net/blog/2019/aug/7/apis-real-life-snagging-global-entry-interview/


# API URL
APPOINTMENTS_URL = "https://ttp.cbp.dhs.gov/schedulerapi/slots?orderBy=soonest&limit=1&locationId={}&minimum=1"

# List of Global Entry locations
LOCATION_IDS = {
    'Tampa': 8020
}

# How often to run this check in seconds
TIME_WAIT = 3600

# Number of days into the future to look for appointments
DAYS_OUT = 60

# Twilio Account Details
# If on trial account, must verify phone number you want to send to.
# You have to buy an SMS-capable number for $1/month to send from.
TWILIO_ACCOUNT_SID = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
TWILIO_AUTH_TOKEN = "**********"

# Digits
# From number has to be purchased first in Twilio console.
# To number has to be verified if on a trial account.
TEXT_TO_NUMBER = "+15555551212"
TEXT_FROM_NUMBER = "+5555551212"

# Dates
now = datetime.now()
future_date = now + timedelta(days=DAYS_OUT)


 "**********"d "**********"e "**********"f "**********"  "**********"s "**********"e "**********"n "**********"d "**********"_ "**********"t "**********"e "**********"x "**********"t "**********"( "**********"t "**********"o "**********"_ "**********"n "**********"u "**********"m "**********"b "**********"e "**********"r "**********", "**********"  "**********"f "**********"r "**********"o "**********"m "**********"_ "**********"n "**********"u "**********"m "**********"b "**********"e "**********"r "**********", "**********"  "**********"m "**********"e "**********"s "**********"s "**********"a "**********"g "**********"e "**********", "**********"  "**********"s "**********"i "**********"d "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********") "**********": "**********"
    client = "**********"

    message = client.messages.create(
        to=to_number, 
        from_=from_number,
        body =message)

    return message.sid

def check_appointments(city, id):
    url = APPOINTMENTS_URL.format(id)
    appointments = requests.get(url).json()
    return appointments

def appointment_in_timeframe(now, future_date, appointment_date):
    if now <= appt_datetime <= future_date:
        return True
    else:
        return False


try:
    while True:
        for city, id in LOCATION_IDS.items():
            try:
                appointments = check_appointments(city, id)
            except Exception as e:
                print("Could not retrieve appointments from API.")
                appointments = []
            if appointments:
                appt_datetime = datetime.strptime(appointments[0]['startTimestamp'], '%Y-%m-%dT%H:%M')
                if appointment_in_timeframe(now, future_date, appt_datetime):
                    message = "{}: Found an appointment at {}!".format(city, appointments[0]['startTimestamp'])
                    try:
                        sms_sid = "**********"
                        print(message, "Sent text successfully! {}".format(sms_sid))
                    except Exception as e:
                        print(e)
                        print(message, "Failed to send text")
                else:
                    print("{}: No appointments during the next {} days.".format(city, DAYS_OUT))
            else:
                print("{}: No appointments during the next {} days.".format(city, DAYS_OUT))
            time.sleep(1)
        time.sleep(TIME_WAIT)
except KeyboardInterrupt:
    sys.exit(0)
