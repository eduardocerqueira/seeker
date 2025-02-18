#date: 2025-02-18T17:08:25Z
#url: https://api.github.com/gists/1f7ea45428078415c56f9b39ba3e644b
#owner: https://api.github.com/users/Kolozuz

from github import Github, Auth
from dotenv import load_dotenv
import os
import csv

load_dotenv()

# Authenticate to Github API using a token
auth = "**********"
gh = Github(auth=auth)

with open("issues.csv", "w",encoding="utf-8", newline="") as issues_csv_file: 
    csv_writer = csv.writer(issues_csv_file)
    csv_writer.writerow(["title","description"])
    
    for issue in gh.get_repo("kolozuz/dev_tools").get_issues(state="open", labels=["Generación de data"], direction="asc"):
        labels = " ".join([f'~"{label.name}"' for label in issue.labels])
        csv_writer.writerow([issue.title, f"{issue.body}\n\n/label {labels}"]){issue.body}\n\n/label {labels}"])