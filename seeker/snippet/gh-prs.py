#date: 2024-11-22T16:58:06Z
#url: https://api.github.com/gists/6e6a77cb7d16cb697cef523ed986085f
#owner: https://api.github.com/users/bmingles

# Map community web ui release versions to enterprise PRs. Useful for updating enterprise version logs.
# Inspired by https://medium.com/@deephavendatalabs/leverage-githubs-awesome-rest-api-f7d34894765b
#
# The script requires some env variables containing username + access tokens for Github + JIRA
# 1. Set env variables in your local environment
# - GH_USERNAME - Your github username
 "**********"# "**********"  "**********"- "**********"  "**********"G "**********"H "**********"_ "**********"T "**********"O "**********"K "**********"E "**********"N "**********"  "**********"- "**********"  "**********"I "**********"n "**********"  "**********"G "**********"i "**********"t "**********"h "**********"u "**********"b "**********", "**********"  "**********"c "**********"r "**********"e "**********"a "**********"t "**********"e "**********"  "**********"a "**********"  "**********"r "**********"e "**********"a "**********"d "**********"- "**********"o "**********"n "**********"l "**********"y "**********"  "**********"f "**********"i "**********"n "**********"e "**********"- "**********"g "**********"r "**********"a "**********"i "**********"n "**********"e "**********"d "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
#   https: "**********"
# - JIRA_USERNAME - Your JIRA username (likely your illumon email address)
 "**********"# "**********"  "**********"- "**********"  "**********"J "**********"I "**********"R "**********"A "**********"_ "**********"T "**********"O "**********"K "**********"E "**********"N "**********"  "**********"- "**********"  "**********"I "**********"n "**********"  "**********"J "**********"I "**********"R "**********"A "**********", "**********"  "**********"c "**********"r "**********"e "**********"a "**********"t "**********"e "**********"  "**********"a "**********"p "**********"i "**********"  "**********"a "**********"c "**********"c "**********"e "**********"s "**********"s "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
#   https: "**********"
#
# 2. Add mapping for env variables to your docker-compose.yml for DH
#    - GH_USERNAME=${GH_USERNAME}
#    - GH_TOKEN= "**********"
#    - JIRA_USERNAME=${JIRA_USERNAME}
#    - JIRA_TOKEN= "**********"
#
#.   NOTE: There are probably other options for loading env variables in DH
#
# 3. Make sure your env variables are active for current shell session, and start your docker container.
#
# Tables
# - minor_releases - Minor release info
# - patch_releases - Patch release info
# - prs - Merged pr info. The `JIRA` column contains a commo separated list of any `DH-####` ids found in the title or body fields.
# - jira - Summary for all `DH-####` tickets found in PR title + body fields.
# - minor_release_prs - Mapping of minor versions to PRs that are included in those releases
# - patch_release_prs - Mapping of minor versions to PRs that are included in those releases

import os, re
os.system("pip install requests")
import requests

from deephaven import DynamicTableWriter
from deephaven import dtypes as dht
from deephaven import time as dhtu

# Github
gh_header = "**********"
org = "deephaven"
repo = "web-client-ui"
api_root = f"https://api.github.com/repos/{org}/{repo}"
pr_url = f"{api_root}/pulls?state=closed&sort=updated&direction=desc&per_page=50"
releases_url = f"{api_root}/releases"

pr_col_defs = {
    "MergedAt": dht.Instant,
    "JIRA": dht.string,
    "BaseRef": dht.string,
    "PR_NUMBER": dht.string,
    "Title": dht.string,
    "Body": dht.string
}
pr_table_writer = DynamicTableWriter(pr_col_defs)

releases_col_defs = {
    "CreatedAt": dht.Instant,
    "Name": dht.string
}
releases_table_writer = DynamicTableWriter(releases_col_defs)

# JIRA
jira_header = "**********"
jira_col_defs = {
    "Issue": dht.string,
    "Summary": dht.string
}
jira_table_writer = DynamicTableWriter(jira_col_defs)

all_ticket_ids = []
prs_json = requests.get(pr_url, auth=gh_header).json()

# Parse PR info into rows
for pr in prs_json:
    # Skip PRs closed without being merged
    if not pr["merged_at"]:
        continue

    # print("merged_at:", dhtu.parse_instant("2020-08-01T12:00:00 ET"), dhtu.parse_instant(pr["merged_at"]))
    merged_at = dhtu.parse_instant(pr["merged_at"])
    base_ref = pr["base"]["ref"]
    pr_number = pr["number"]
    # pr_url = f"https://github.com/deephaven/web-client-ui/pull/{pr_number}"
    title = pr["title"]
    body = ""
    if pr["body"]:
        body = pr["body"]

    # Distinct JIRA ids found in title + body fields
    title_ids = re.findall(r'DH-\d+', title)
    body_ids = re.findall(r'DH-\d+', body)
    ids = set(title_ids + body_ids)
    jira_id_str = ', '.join(ids)

    all_ticket_ids.extend(ids)

    pr_table_writer.write_row(merged_at, jira_id_str, base_ref, str(pr_number), title, body)

releases_json = requests.get(releases_url, auth=gh_header).json()
for release in releases_json:
    created_at = dhtu.parse_instant(release["created_at"])
    name = release["name"]
    releases_table_writer.write_row(created_at, name)

# Get JIRA issue descriptions
for key in set(all_ticket_ids):
    jira_url = f"https://deephaven.atlassian.net/rest/api/2/issue/{key}?fields=summary"
    ticket = requests.get(jira_url, auth=jira_header).json()
    summary = ticket["fields"]["summary"]
    jira_table_writer.write_row(key, summary)

# exposed tables
minor_releases = releases_table_writer.table.where(filters=["Name.endsWith(`.0`)"])
patch_releases = releases_table_writer.table.where(filters=["!Name.endsWith(`.0`)"])
prs = pr_table_writer.table
jira = jira_table_writer.table

minor_release_prs = prs\
    .where(filters=["BaseRef = `main`"])\
    .raj(table=minor_releases, on=["MergedAt <= CreatedAt"])\
    .move_columns(idx=0, cols=["Name", "CreatedAt"])\
    .rename_columns(cols=["ReleasedAt = CreatedAt"])

patch_release_prs = prs\
    .where(filters=["BaseRef != `main`"])\
    .raj(table=patch_releases, on=["MergedAt <= CreatedAt"])\
    .move_columns(idx=0, cols=["Name", "CreatedAt"])\
    .rename_columns(cols=["ReleasedAt = CreatedAt"])