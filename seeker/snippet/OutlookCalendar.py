#date: 2024-01-22T17:04:46Z
#url: https://api.github.com/gists/c412be7cad5c201ce4205a3ab77431f4
#owner: https://api.github.com/users/aaronkirkman

from win32com.client import Dispatch
from tabulate import tabulate
import datetime
import pdb

OUTLOOK_FORMAT = '%m/%d/%Y %H:%M'
outlook = Dispatch("Outlook.Application")
ns = outlook.GetNamespace("MAPI")

appointments = ns.GetDefaultFolder(9).Items 

# Restrict to items in the next 30 days (using Python 3.3 - might be slightly different for 2.7)
begin = datetime.date.today()
end = begin + datetime.timedelta(days = 30);
restriction = "[Start] >= '" + begin.strftime("%m/%d/%Y") + "' AND [End] <= '" +end.strftime("%m/%d/%Y") + "'"
restrictedItems = appointments.Restrict(restriction)

appointments.Sort("[Duration]")
appointments.IncludeRecurrences = "True"

# Iterate through restricted AppointmentItems and print them
calcTableHeader = ['Title', 'Organizer', 'Start', 'Duration(Minutes)'];
calcTableBody = [];

#pdb.set_trace()
for appointmentItem in appointments:
    row = []
    row.append(appointmentItem.Subject)
    row.append(appointmentItem.Organizer)
    row.append(appointmentItem.Start.Format(OUTLOOK_FORMAT))
    row.append(appointmentItem.Duration)
    calcTableBody.append(row)


print tabulate(calcTableBody, headers=calcTableHeader);