#date: 2024-01-25T17:01:05Z
#url: https://api.github.com/gists/3a75ee820570914c7f19bfc9d15aafd9
#owner: https://api.github.com/users/kaihendry

name=$1
value=$2
aws ssm put-parameter \
        --name "${name}" --value "${value}" --type "SecureString"