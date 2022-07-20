#date: 2022-07-20T17:03:14Z
#url: https://api.github.com/gists/9256da09be02a249454cac83cbcf0150
#owner: https://api.github.com/users/Fraasi

#!/usr/bin/env python

"""Simple (unix like) command line calendar in python"""

import os
from datetime import datetime
import argparse
import calendar


currentMonth = datetime.now().month
currentYear = datetime.now().year

parser = argparse.ArgumentParser(description='Simple command line calendar')
parser.add_argument('month', help='what month to display (1-12)',
                    nargs='?', default=currentMonth, type=int, choices=range(1, 13), metavar='month')
parser.add_argument('year', help='what year to display',
                    nargs='?', default=currentYear, type=int)
parser.add_argument('-s', help='display sunday as the first day of the week (default: monday)', default=False, action='store_true')
group = parser.add_mutually_exclusive_group()
group.add_argument('-y', help='display whole year', action='store_true')
group.add_argument(
    '-M', help='display previous, current and next month', action='store_true')
args = parser.parse_args()

if args.s:
    calendar.setfirstweekday(6)
if args.y:
    cal = calendar.calendar(args.year)
    print(cal, end='')
elif args.M:
    if args.month == 1:
        prevMonth = 12
        prevYear = args.year - 1
    else:
        prevMonth = args.month - 1
        prevYear = args.year
    if args.month == 12:
        nextMonth = 1
        nextYear = args.year + 1
    else:
        nextMonth = args.month + 1
        nextYear = args.year

    prevWeeks = calendar.monthcalendar(prevYear, prevMonth)
    currWeeks = calendar.monthcalendar(args.year, args.month)
    nextWeeks = calendar.monthcalendar(nextYear, nextMonth)
    prevTitle = f'{calendar.month_name[prevMonth]} {str(prevYear)}'
    currTitle = f'{calendar.month_name[args.month]} {str(args.year)}'
    nextTitle = f'{calendar.month_name[nextMonth]} {str(nextYear)}'
    daynames = calendar.weekheader(2)
    print(f'{prevTitle:^20}    {currTitle:^20}    {nextTitle:^20}')
    print(f'{daynames:^20}    {daynames:^20}    {daynames:^20}')
    # months can have different week lengths eg. [5, 6, 5]...
    # append 7 empty spaces to shorter ones to print right
    maxlength = max([len(prevWeeks), len(currWeeks), len(nextWeeks)])
    for arr in [prevWeeks, currWeeks, nextWeeks]:
        arr.append(' '*7) if len(arr) < maxlength else None
    for i in range(maxlength):
        prevs = [' ' if n == 0 else n for n in prevWeeks[i]]
        currs = [' ' if n == 0 else n for n in currWeeks[i]]
        nexts = [' ' if n == 0 else n for n in nextWeeks[i]]
        print("{:>2} {:>2} {:>2} {:>2} {:>2} {:>2} {:>2}    {:>2} {:>2} {:>2} {:>2} {:>2} {:>2} {:>2}    {:>2} {:>2} {:>2} {:>2} {:>2} {:>2} {:>2}".format(*prevs, *currs, *nexts))
else:
    os.system('') # ANSI codes doesnt work without this hack, https://github.com/python/cpython/issues/84315

    cal = calendar.month(args.year, args.month)

    if args.month == currentMonth and args.year == currentYear:
        currentDay = datetime.now().day
        splt = cal.rsplit(str(currentDay), 1)
        cal = f'\033[00;36m{currentDay}\033[0m'.join(splt)

    print(cal, end='')
