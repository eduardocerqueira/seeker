#date: 2022-03-03T16:51:17Z
#url: https://api.github.com/gists/625aa30ecaa80f3c5fb6c22f256e5161
#owner: https://api.github.com/users/bigbaer01

"""
    File name:      call_analytics_usa_gov_api.py
    Author:         Randy Runtsch
    Date:           March 23, 2021
    Description:    Controller for calls to obtain report from the c_analytics_usa_gov_api class.
"""

from c_analytics_usa_gov_api import c_analytics_usa_gov_api

# Call c_analytics_usa_gov_api with the name of the report to retrieve and the name of the 
# JSON file to store it in.

print("Program started.")

c_analytics_usa_gov_api('os-browser', 'D:\project_data/analytics_usa_gov/os_browser_report.json')

print("Program completed")
