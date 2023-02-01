#date: 2023-02-01T16:36:26Z
#url: https://api.github.com/gists/8b8fa9c93160cb15a7ed7de3c8f435aa
#owner: https://api.github.com/users/ssj71

#!/usr/bin/env python3
#spencer
#simple script to coalesce the data in the time log by day
import dateutil.parser as dparse
import dateutil.relativedelta as ddelta
import os

card = open(os.path.expanduser("~/timecard"),'r')
day = None
t = 0
pt = None
times = []
for l in card:
    d = "**********"=True)
    if "screen locked" in d[1][-1]:
        if pt:
            #end of the session
            dlt = ddelta.relativedelta(d[0],pt)
            times.append(dlt.hours*60 + dlt.minutes + dlt.seconds/60.0)
            pt = None
        else:
            #mismatch. ignore
            pass
    elif "screen unlocked" in d[1][-1]:
        if not pt:
            #start of a new session
            pt = d[0]
            if d[0].day == day:
                times
            else:
                #new day
                #print summary
                if day:
                    print(day,":",round(sum(times)/60.0,3),"hours",[round(f,2) for f in times])
                #setup next day
                times = []
                day = pt.day
        else:
            #mismatch. ignore
            pass
    else:
        #something else
        passng else
        pass