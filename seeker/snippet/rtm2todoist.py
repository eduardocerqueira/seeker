#date: 2022-03-08T16:50:31Z
#url: https://api.github.com/gists/68c84868098dcaa04e44522bebb77638
#owner: https://api.github.com/users/notconfusing

#!/usr/bin/env python
#
# Converting an ICS file produced by Remember the milk to a CSV file
# that can be read by Todoist.
#
# Bye-bye, remember the milk. Raising the annual price from $25.00 to $39.99,
# while stagnating with the old interface, is the way to lose your loyal pro users.
#
# This script requires icalendar, which can be installed like this:
#
#   pip install icalendar
#
# A great manual on exporting RTM tasks and importing them to Todoist
# can be found here; this script automates the last step ICS-to-CSV:
#
#  http://martin1963projects.blogspot.co.at/2016/03/migrating-tasks-from-rtm-remember-milk.html
#
#
# I debugged the script on 260 of my own RTM tasks, but it is certainly incomplete.
# The recurring tasks may require some attention and manual hacking,
# as Todoist does not support all of the complex repetition rules of RTM.
# RTM notes cannot be imported/exported, they need manual efforts.
#
# ----------------------------------------------------------------------------
# "THE BEER-WARE LICENSE" (Revision 42.5):
# <konnov@forsyte.at> wrote this file. As long as you retain this notice you
# can do whatever you want with this stuff, including modification.
# If we meet some day, and you think # this stuff is worth it, you can buy me
# a beer in return. Igor Konnov, 2018.
# ----------------------------------------------------------------------------

import csv
from icalendar import Calendar
import re
import sys

# default settings, modify if needed
ENCODING = 'utf-8'
AUTHOR = 'your.username (your.id)'
                 # the author, find out your id by exporting a todoist task to CSV
DATE_LANG = 'en' # the language in which the dates are written
TODOIST_HEADER = ['TYPE', 'CONTENT', 'PRIORITY', 'INDENT', 'AUTHOR',
                  'RESPONSIBLE', 'DATE', 'DATE_LANG', 'TIMEZONE']

WEEKDAY_RULE = re.compile('(?P<signal>[+-]?)(?P<relative>[\d]?)'
                          '(?P<weekday>[\w]{2})$') # the pattern from iCalendar
DATE_RE = re.compile('^(?P<year>[\d]{4})(?P<month>[\d]{2})(?P<day>[\d]{2})$')
DATETIME_RE = re.compile('^(?P<year>[\d]{4})(?P<month>[\d]{2})(?P<day>[\d]{2})'
                         'T(?P<hour>[\d]{2})(?P<min>[\d]{2})(?P<sec>[\d]{2})(Z?)$')

DAY_DICT = { "SU": "sunday", "MO": "monday", "TU": "tuesday",
             "WE": "wednesday", "TH": "thursday", "FR": "friday",
             "SA": "saturday" }

FREQ_DICT = { "SECONDLY": "second", "MINUTELY": "minute", "HOURLY": "hour",
              "DAILY": "day", "WEEKLY": "week", "MONTHLY": "month",
              "YEARLY": "year" }

MONTHS = ["Nullary", "January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]


def parse_args():
    argv = sys.argv[1:]
    if len(argv) != 2:
        print "Use: %s in.ics out.csv" % sys.argv[0]
        sys.exit(1)

    return argv


def translate_todo(writer, entry):
    """
    Translate a single VTODO entry to a CSV line.

    Arguments:
    writer -- a CSV writer,
    entry -- an iCalendar entry (an entry different from VTODO is ignored)
    """
    if entry.name == 'VTODO':
        row = {}
        row['TYPE'] = 'task'
        row['CONTENT'] = entry['summary']
        row['AUTHOR'] = AUTHOR
        row['RESPONSIBLE'] = ''
        row['PRIORITY'] = entry['priority'] if 'priority' in entry else "4"
        row['INDENT'] = "1"
        row['DATE'] = ''
        row['DATE_LANG'] = DATE_LANG
        row['TIMEZONE'] = convert_tz(entry['due']) if 'due' in entry else ''
        if 'due' in entry:
            due_date = convert_due(entry['due'])
            if 'rrule' in entry:
                recur = convert_recurrence(entry['summary'], entry['rrule'])
                due_date = "%s starting %s" % (recur, due_date)

            row['DATE'] = due_date

        writer.writerow(row)
        return 1
    else:
        return 0


def convert_tz(due):
    p = due.params
    return p['TZID'] if 'TZID' in due.params else ''


def convert_due(due):
    """
    Convert a due date.
    """
    dts = due.to_ical()
    dtm = DATETIME_RE.match(dts)
    dm = DATE_RE.match(dts)
    if dtm:
        # If there is a date and time.
        # The trick is that Todoist only parses human-readable dates.
        d = dtm.groupdict()
        year, month, day = d['year'], d['month'], d['day']
        hour, minute = d['hour'], d['min']
        return "%d %s %d at %s:%s" \
                % (int(day), MONTHS[int(month)], int(year), hour, minute)
    elif dm:
        # If there is a date only. A machine-readable date would also work,
        # but we make a human-readable one here as well.
        d = dm.groupdict()
        year, month, day = d['year'], d['month'], d['day']
        return "%d %s %d" % (int(day), MONTHS[int(month)], int(year))
    else:
        # Fall back to the original date
        return dts


def convert_recurrence(content, rr):
    """
    Convert a recurrence rule in the todoist format.
    As todoist's language is limited, this translation is incomplete.
    """
    if 'freq' not in rr or len(rr['freq']) != 1:
        # too many frequencies
        print 'Cannot parse recurrence rule: %s' % rr.to_ical()
        print 'In todo: %s' % content
        return ""

    freq = FREQ_DICT[rr['freq'][0]]
    if 'interval' in rr:
        # how often the repetition occurs
        interval = rr['interval'][0] # more than one interval?
        if 'byday' in rr: # on which days
            return "every " + convert_by_day(rr, content, freq, interval)
        else:
            return "every %d %s%s" % (interval, freq, ("s" if interval > 1 else ""))
    else:
        return "every " + freq


def convert_by_day(rr, content, freq, interval):
    """
    Convert a BYDAY rule that can appear in monthly and yearly
    recurrence rules. The translation cannot be done precisely as Todoist
    does not seem to support rules like 'every 2 months on the last Sunday'.
    """
    # FREQ=WEEKLY;WKST=MO;INTERVAL=1;BYDAY=SU
    # FREQ=MONTHLY;WKST=SU;INTERVAL=3;BYDAY=2SA
    if freq == 'week':
        return ",".join([DAY_DICT[d] for d in rr['byday']])
    elif freq == 'month':
        days = []
        for dd in rr['byday']:
            # iCalendar has already parsed the day spec, so it should work
            match = WEEKDAY_RULE.match(dd).groupdict()
            signal = match['signal'] if match['signal'] else '+'
            weekday = match['weekday']
            relative_str = match['relative']
            assert(relative_str) # it should be like that with MONTHLY
            relative = int(relative_str) if signal == '+' else -int(relative_str)

            if relative > 0 and interval == 1:
                days.append(DAY_DICT[weekday])
            elif relative == -1 and interval == 1:
                days.append("last " + DAY_DICT[weekday])
            else:
                print "Not supported by todoist: %s" % rr.to_ical()
                print 'In todo: %s' % content
                if relative < 0:
                    count = interval * 4 + relative
                else:
                    count = (interval - 1) * 4 + relative

                text = "%d %s" % (count, DAY_DICT[weekday])
                print 'Approximated as: %s' % text
                days.append(text)

        return ",".join(days)
    else:
        print "Not supported by todoist: %s" % rr.to_ical()
        print 'In todo: %s' % content
        print 'IGNORED'
        return ""


# main
if __name__ == "__main__":
    iname, oname = parse_args()
    with open(iname, 'r') as inf:
        cal = Calendar().from_ical(inf.read().decode(ENCODING), multiple=False)
        prodid = cal['PRODID']
        print 'Imported calendar by %s' % prodid
        if prodid.find('Remember The Milk') < 0:
            print 'WARNING: this calendar appears not to be exported by RTM'

        with open(oname, 'wb') as csvf:
            csvwriter = csv.DictWriter(csvf, delimiter=',', fieldnames=TODOIST_HEADER)
            csvwriter.writeheader()
            ntodos = 0
            for ev in cal.subcomponents:
                ntodos += translate_todo(csvwriter, ev)
            
            print ''
            print 'Converted %d todo entries.' % ntodos
            print ''
            print 'Import %s as a template in Todoist inbox and'\
                ' assign the projects.' % oname
            print 'The notes cannot be converted automatically. Copy them manually.'
            print ''
