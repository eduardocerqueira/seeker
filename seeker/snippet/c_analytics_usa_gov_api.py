#date: 2022-03-03T16:51:17Z
#url: https://api.github.com/gists/625aa30ecaa80f3c5fb6c22f256e5161
#owner: https://api.github.com/users/bigbaer01

import requests
import json

class c_analytics_usa_gov_api:
    
    """
        File name:      c_analytics_usa_gov_api.py
        Class name:     c_analytics_usa_gov
        Author:         Randy Runtsch
        Date:           March 23, 2021
        Description:    Call the analytics.usa.gov API with a query and handle the results.
    """

    def __init__(self, report, json_file_nm):

        # Open the output JSON file, get the report from api.gsa.gov, and close the output file.


        # IMPORTANT: Visit https://open.gsa.gov/api/dap/ to request a personal API key and insert it between the quotes:
        self.API_KEY = "INSERT_YOUR_PERSONAL_API_KEY_HERE"

        json_file = open(json_file_nm, 'w', encoding='utf-8')
        self.get_report(report, json_file)
        json_file.close()

    def get_report(self, report, json_file):

        # Call the API to get the report. Write it to a JSON file.

        response = requests.get('https://api.gsa.gov/analytics/dap/v1.1/reports/' + report + '/data?api_key=' + self.API_KEY)

        if response.status_code != 200:
            # Something went wrong.
            raise ApiError('GET /tasks/ {}'.format(resp.status_code))
       
        json.dump(response.json(), json_file, indent = 6)




