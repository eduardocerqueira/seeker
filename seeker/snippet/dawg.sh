#date: 2025-03-05T16:58:43Z
#url: https://api.github.com/gists/8f7614a88aa94ed008f60b3fe2213e54
#owner: https://api.github.com/users/RajChowdhury240

aws cloudformation list-stack-instances --stack-set-name <YourStackSetName> --query 'Summaries[?DriftStatus==`DRIFTED`].[Account, Region]' --output table