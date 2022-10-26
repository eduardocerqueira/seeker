#date: 2022-10-26T17:02:43Z
#url: https://api.github.com/gists/0f1538ebc8268a88cd4e0a0a61445287
#owner: https://api.github.com/users/ianmcook

from jira import JIRA
from github import Github
import re
import pandas as pd

jira = JIRA('https://issues.apache.org/jira')
github = Github("ENTER_YOUR_GITHUB_PAT_HERE")
repo = github.get_repo("apache/arrow")

jira_issues = []
i = 0
chunk_size = 200
while True:
    chunk = jira.search_issues(f'project = ARROW AND status = Resolved ORDER BY key', startAt=i, maxResults=chunk_size, fields='assignee,comment')
    i += chunk_size
    jira_issues += chunk.iterable
    if i >= chunk.total:
        break

jira_issues = jira_issues_all[8:]
#jira_issues_all = jira_issues
#jira_issues = jira_issues_all[8764:]

jira_users = []
gh_users = []

# iterate over the issues
for jira_issue in jira_issues:
    print(jira_issue.key)
    if jira_issue.fields.assignee is None:
        continue
    n = len(jira_issue.fields.comment.comments)
    for j in reversed(range(0, n)):
        t = jira_issue.fields.comment.comments[j].body
        s = re.search(r"^Issue resolved by pull request ([0-9]{1,5})\n\[https://github.com/apache/arrow/pull/\1]$", t)
        if s:
            gh_issue_id = int(s.group(1))
            gh_issue = repo.get_issue(number=gh_issue_id)
            print('\t' + jira_issue.fields.assignee.key)
            jira_users.append(jira_issue.fields.assignee.key)
            print('\t' + gh_issue.user.login)
            gh_users.append(gh_issue.user.login)
            break

df = pd.DataFrame({'jira': jira_users, 'github': gh_users})
df.to_csv('dirty.csv')
