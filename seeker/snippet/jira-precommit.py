#date: 2023-10-27T16:53:30Z
#url: https://api.github.com/gists/a1c85eb90b5cd5da63a79460cb461cbb
#owner: https://api.github.com/users/RoseSecurity

#!/usr/bin/env python3

from jira import JIRA
import subprocess

# Set your Jira server URL, Email, and API token
email = ""
server_url = ""
api_token = "**********"

# Initialize the Jira client
jira = JIRA(server=server_url, options={'server': "**********"

# Function to capture the last commit message
def capture_last_commit_message():
    try:
        git_log = subprocess.check_output(['git', 'log', '-1', '--pretty=format:%s'], stderr=subprocess.STDOUT)
        return git_log.decode('utf-8')
    except subprocess.CalledProcessError as e:
        return f"Error: {e.returncode}\n{e.output.decode('utf-8')}"

def extract_project_key(commit_message):
    words = commit_message.strip().split()
    if words:
        project_key = words[0]
        commit_title = ' '.join(words[1:])
        return project_key, commit_title
    else:
        return None, None

commit_message = capture_last_commit_message()
project_key, commit_title = extract_project_key(commit_message)

try:
    # First word of the commit message is the project (Ex: "BUG Fix Broken Dependencies" would create in the BUG project)
    project = jira.project(project_key)

    # Create the issue in the project
    issue_dict = {
        'project': {'key': project_key},
        'summary': commit_title,
        'description': commit_title,
        'issuetype': {'name': 'Task'},  # You can adjust the issue type as needed (Epic, Task, etc.)
    }
    new_issue = jira.create_issue(fields=issue_dict)

    print(f"Issue '{new_issue.key}' created successfully")

except Exception as e:
    print(f"An error occurred: {str(e)}")
exit(0)d: {str(e)}")
exit(0)