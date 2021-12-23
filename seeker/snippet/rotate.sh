#date: 2021-12-23T16:56:43Z
#url: https://api.github.com/gists/3f61ab481955af558f1e316f1be7a66a
#owner: https://api.github.com/users/jlamoree

#!/usr/bin/env bash

profile="aws_cli_profile_name"
directory_id="d-1234567890"
username_pattern="user"
password_title="AWS Directory User"

for n in `seq -f "%02g" 1 5`; do
  uuid=$(op create item Password --generate-password --title "$password_title $n" | jq -r .uuid)
  password=$(op get item $uuid --fields password)
  aws --profile $profile ds reset-user-password --directory-id $directory_id \
    --user-name "${username_pattern}$n" --new-password "$password"
  sleep 5
done