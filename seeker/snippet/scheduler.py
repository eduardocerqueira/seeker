#date: 2021-10-25T17:12:09Z
#url: https://api.github.com/gists/87184bcf16619c4a5c437042252847fd
#owner: https://api.github.com/users/poojitagarg

from script import *
from datetime import datetime
from datetime import date
from datetime import time
import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
import gspread
from oauth2client.service_account import ServiceAccountCredentials

s=['https://www.googleapis.com/auth/spreadsheets',
'https://www.googleapis.com/auth/drive']

dt=date.today().strftime('%d/%m/%Y')
now_date=datetime.strptime(dt,'%d/%m/%Y')
rem_day=now_date.day
rem_month=now_date.month
rem_year=now_date.year

t=datetime(rem_year,rem_month,rem_day,23,30)
local = pytz.timezone("Asia/Kolkata")
local_dt = local.localize(t, is_dst=None)
utc_dt = local_dt.astimezone(pytz.utc)


scheduler = BlockingScheduler()
creds= ServiceAccountCredentials.from_json_keyfile_name("credentials.json",s)
client=gspread.authorize(creds)

sheet = client.open("Reminders").sheet1
list_of_lists = sheet.get_all_values()
print(list_of_lists)
for row in list_of_lists:
    if row[0] == dt:
        scheduler.add_job(send_rem, 'date', run_date=utc_dt, args=[row[0],row[1]])
        print("yoyo")
    else:
        pass
        
scheduler.start()


