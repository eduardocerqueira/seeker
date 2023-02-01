#date: 2023-01-31T17:08:26Z
#url: https://api.github.com/gists/a254aef15efc0e10ca253da58d8fb849
#owner: https://api.github.com/users/gducharme

#/bin/bash

gh auth login --with-token < token.txt

#AUTHOR is the github name
#JIRA_ID is on the tin
#GH_TOKEN
#JIRA_TOKEN



curl -H "Authorization: "**********"
 -s "https://api.github.com/search/issues?q=type:pr+author:$AUTHOR+user:SpringCare+created:>=2022-07-01" | jq '.total_count'
curl -H "Authorization: "**********"
-s "https://api.github.com/search/issues?q=is:pr+review-requested:$AUTHOR+user:SpringCare+created:>=2022-07-01" | jq '.total_count'

#Get all PRs and reviews from 22-01-01
curl -H "Authorization: "**********"
  -s "https://api.github.com/search/issues?q=type:pr+author:$AUTHOR+user:SpringCare+created:>=2022-01-01" | jq '.total_count'
curl -H "Authorization: "**********"
  -s "https://api.github.com/search/issues?q=is:pr+review-requested:$AUTHOR+user:SpringCare+created:>=2022-01-01" | jq '.total_count'

#Get all Jira tickets from 22-07-01 Done/Wont fix
curl -X GET "https://springhealth.atlassian.net/rest/api/2/search?jql=assignee%20in%20($JIRA_ID)%20AND%20createdDate%3E%3D%202022-07-01%20AND%20status%20IN%20(Done%2C%20WONTFIX%2C%20Released%2C%20Closed)%20ORDER%20BY%20status%20DESC" \
-u "geoffrey.ducharme@springhealth.com: "**********"
curl -X GET "https://springhealth.atlassian.net/rest/api/2/search?jql=assignee%20in%20($JIRA_ID)%20AND%20createdDate%3E%3D%202022-07-01%20ORDER%20BY%20status%20DESC" \
-u "geoffrey.ducharme@springhealth.com: "**********"


#Get all Jira tickets from 22-01-01 Done/Wont fix
curl -X GET "https://springhealth.atlassian.net/rest/api/2/search?jql=assignee%20in%20($JIRA_ID)%20AND%20createdDate%3E%3D%202022-01-01%20AND%20status%20IN%20(Done%2C%20WONTFIX%2C%20Released%2C%20Closed)%20ORDER%20BY%20status%20DESC" \
-u "geoffrey.ducharme@springhealth.com: "**********"

curl -X GET "https://springhealth.atlassian.net/rest/api/2/search?jql=assignee%20in%20($JIRA_ID)%20AND%20createdDate%3E%3D%202022-01-01%20ORDER%20BY%20status%20DESC" \
-u "geoffrey.ducharme@springhealth.com: "**********"atus%20DESC" \
-u "geoffrey.ducharme@springhealth.com: "**********"| jq '.total'