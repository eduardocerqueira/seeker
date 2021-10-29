#date: 2021-10-29T16:42:45Z
#url: https://api.github.com/gists/1c8b53bbd29f4fdd4f2b35c31dec3631
#owner: https://api.github.com/users/alexp-dmpg

import json
from apiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials


SCOPES = ["https://www.googleapis.com/auth/analytics.readonly"]
# add service account json data here
KEY_FILE_LOCATION = "INSERT_PATH_HERE"
# add view id from GA here
VIEW_ID = "INSERT_ID_HERE"


def initialize_analyticsreporting():
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        KEY_FILE_LOCATION, SCOPES
    )
    analytics = build("analyticsreporting", "v4", credentials=credentials)
    return analytics


def get_report(analytics):
    return (
        analytics.reports()
        .batchGet(
            body={
                "reportRequests": [
                    {
                        "viewId": VIEW_ID,
                        # date range will provide a rolling report of data from previous day
                        "dateRanges": [{"startDate": "yesterday", "endDate": "yesterday"}],
                        # get the total events metric
                        "metrics": [{"expression": "ga:totalEvents"}],
                        # break that metric down by page and event label combined
                        "dimensions": [
                            {"name": "ga:eventLabel"},
                            {"name": "ga:pagePath"},
                        ],
                        # here we can add as many filters as we like
                        "dimensionFilterClauses": [
                            {
                                "operator": "AND",
                                "filters": [
                                    # removing all event labels with no data
                                    {
                                        "dimensionName": "ga:eventLabel",
                                        "not": True,
                                        "operator": "EXACT",
                                        "expressions": ["(not set)"],
                                        "caseSensitive": False,
                                    },
                                    # filtering our data for only certain pages
                                    {
                                        "dimensionName": "ga:pagePath",
                                        "not": False,
                                        "operator": "BEGINS_WITH",
                                        "expressions": ["/dashboard"],
                                        "caseSensitive": False,
                                    },
                                ],
                            }
                        ],
                    }
                ]
            }
        )
        .execute()
    )


def store_response(response):
    for ind, report in enumerate(response.get("reports", [])):
        # loop through each returned report and identify what data each column is holding
        column_header = report.get('columnHeader', {})
        dims_list = column_header.get("dimensions", [])
        metrs_raw = column_header.get("metricHeader", {}).get("metricHeaderEntries", [])
        metrs_list = list(map(lambda x: x["name"], metrs_raw))

        page_ind = dims_list.index("ga:pagePath")
        labl_ind = dims_list.index("ga:eventLabel")
        evts_ind = metrs_list.index("ga:totalEvents")

        output_dict, output_dict_sorted = {}, {}

        for row in report.get('data', {}).get('rows', []):
            # loop through each data row, converting the raw results into a
            # list of event labels with event totals, segmented by each page
            row_dims = row.get('dimensions', [])
            row_metrs = row.get('metrics', [])

            output_dict[row_dims[page_ind]] = output_dict.get(row_dims[page_ind], {})
            output_dict[row_dims[page_ind]][row_dims[labl_ind]] = int(row_metrs[evts_ind].get("values", [])[0])

        for entry in output_dict:
            # make it more readable by sorted event totals in descending order for each page
            output_dict_sorted[entry] = dict(sorted(output_dict[entry].items(), key=lambda x: x[1], reverse=True))

        with open(("output_%s.json" % ind), "w") as file:
            # save each formatted report output as json
            file.write(json.dumps(output_dict_sorted))


def main():
    analytics = initialize_analyticsreporting()
    response = get_report(analytics)
    store_response(response)


if __name__ == "__main__":
    main()
