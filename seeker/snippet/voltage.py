#date: 2024-01-24T16:55:01Z
#url: https://api.github.com/gists/df9a561ca44ac286aaf628d421ab6617
#owner: https://api.github.com/users/vizistaha

# -*- coding: utf-8 -*-
import csv
import time
import requests
import schedule
from bs4 import BeautifulSoup

def voltage_check():
    # access apcupsd
    apcupsd = "http://10.0.1.5/apcupsd/upsfstats.cgi?host=127.0.0.1"
    try:
        apcupsd_respon = requests.get(apcupsd, timeout=5)
        if apcupsd_respon.status_code == 200:
            html = apcupsd_respon.content
            apcupsd_respon.close()
            # BeautifulSoup
            batterysoup = BeautifulSoup(html,"html.parser")
            dataframe = batterysoup.find("pre").text
            data = dataframe.split("\n")
            # check time
            check_time = data[1]
            # line voltage
            voltage = data[11]
            # Save
            with open("voltage_record.csv", mode="a", newline="") as tape:
                recording=csv.writer(tape)
                recording.writerow([check_time,voltage])
                tape.flush()
                print("logging successfully")
                tape.close()
        else:
            apcupsd_respon.close()
            print(apcupsd_respon.status_code)
    except requests.exceptions.Timeout:
        print("Timeout")
    except Exception:
        print("Error")

# Execute setting
schedule.every(15).minutes.do(voltage_check)
# Running Loop
print("Now monitoring voltage, pressing CTRL+C to exit.")
try:
    while True:
        schedule.run_pending()
        time.sleep(1)
except Exception as error_status:
    print(error_status)
except KeyboardInterrupt:
    print("Thank you for using the voltage monitor.\r\nGoodBye ...")