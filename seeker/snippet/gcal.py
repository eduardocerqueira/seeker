#date: 2023-08-01T16:40:52Z
#url: https://api.github.com/gists/9bb1d0500874f50792df734047bb03bb
#owner: https://api.github.com/users/vikrum

# MIT License
# 
# Copyright (c) April 2023 Vikrum Nijjar
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import dateparser
from datetime import timedelta
from urllib.parse import urlencode

#Google Calendar Link format:
#    https://github.com/InteractionDesignFoundation/add-event-to-calendar-docs/blob/main/services/google.md
#    https://github.com/AnandChowdhary/calendar-link/blob/master/src/index.ts

# check if datetime is midnight
def is_midnight(datetime):
    return datetime.hour == 0 and datetime.minute == 0 # and datetime.second == 0

# check if datetime is 11:59pm
def is_1159(datetime):
    return datetime.hour == 23 and datetime.minute == 59 # and datetime.second == 0

# format datetime as Month, Day, Year
def gcal_date_format_datetime_human_friendly(datetime):
    if is_midnight(datetime):
        return datetime.strftime("%B %d, %Y")
    else:
        return datetime.strftime("%B %d, %Y %l:%M%p")

# format datetime as YYYYMMDDTHHmmSS
def gcal_date_time_format_datetime(datetime):
    return datetime.strftime("%Y%m%dT%H%M%S")

# format datetime as YYYYMMDD
def gcal_date_format_datetime(datetime):
    return datetime.strftime("%Y%m%d")

def convert_rawdate_to_gcal_format(rawdate):
    datescomponents = []
    if ' ' in rawdate:
        splitdate = rawdate.split(' ')
        try:
            start = dateparser.parse(splitdate[0])
            end = dateparser.parse(splitdate[len(splitdate) - 1])

            #if the LLM returns 11:59pm, increment to midnight to trigger 'all day' events in GCal
            if(is_midnight(start) and is_1159(end)):
                end = end + timedelta(minutes=1)
                datescomponents.append(gcal_date_format_datetime(start))
                datescomponents.append(gcal_date_format_datetime(end))
            else:
                datescomponents.append(gcal_date_time_format_datetime(start))
                datescomponents.append(gcal_date_time_format_datetime(end))
        
        except:
            # this will definitely break the gcal link, but it's better than nothing
            datescomponents.append(splitdate[0])
            datescomponents.append(splitdate[len(splitdate) - 1])
    else:
        try:
            start = dateparser.parse(rawdate)
            if(is_midnight(start)): 
                # infer that this is an "all day" event, this gets formatted as YYYYMMDD with +1 day
                end = start + timedelta(days=1)
                datescomponents.append(gcal_date_format_datetime(start))
                datescomponents.append(gcal_date_format_datetime(end))

            else:
                end = start + timedelta(hours=1)
                datescomponents.append(gcal_date_time_format_datetime(start))
                datescomponents.append(gcal_date_time_format_datetime(end))

        except:
            datescomponents.append(rawdate)
            datescomponents.append(rawdate)

    return datescomponents
    

def convert_rawdate_to_human_friendly_start_only(rawdate):
    result = rawdate
    if ' ' in rawdate:
        splitdate = rawdate.split(' ')
        try:
            start = dateparser.parse(splitdate[0])
            result = gcal_date_format_datetime_human_friendly(start)
        except:
            result = rawdate
    else:
        try:
            start = dateparser.parse(rawdate)
            result = gcal_date_format_datetime_human_friendly(start)

        except:
            result = rawdate
    return result

# https://calendar.google.com/calendar/render?action=TEMPLATE
#   &text=Birthday
#   &dates=20201231T193000Z/20201231T223000Z
#   &details=With%20clowns%20and%20stuff
#   &location=North%20Pole

def make_google_calendar_link(rawdate, description, note):
    datescomponents = convert_rawdate_to_gcal_format(rawdate)
    baseurl = "https://calendar.google.com/calendar/render"
    
    query_params = {
     "action": "TEMPLATE",
     "text": description,
     "details": note,
     "dates": f"{datescomponents[0]}/{datescomponents[1]}"
    }   

    # Use urlencode to generate the URL-encoded query string
    query_string = urlencode(query_params)

    # Combine the base URL with the query string
    return baseurl + '?' + query_string