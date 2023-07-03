#date: 2023-07-03T16:36:00Z
#url: https://api.github.com/gists/c78ba6008f58b02b9d2129d47876d88d
#owner: https://api.github.com/users/SYNchroACK

from pycti import OpenCTIApiClient

api_url = "http://opencti:8080"
api_token = "**********"

api = "**********"

for report in api.report.list():
    for file_path in report["importFilesIds"]:
        if "pdf" not in file_path:
            continue

        name = report["name"]
        description = report["description"]
        pdf = f"{api_url}/storage/view/{file_path}"

        print(f"\n\n# {name}\n{description}\n\n{pdf}")
n}\n\n{pdf}")
