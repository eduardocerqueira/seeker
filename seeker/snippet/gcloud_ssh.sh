#date: 2023-03-28T16:50:08Z
#url: https://api.github.com/gists/860311aa29560e0cb1df3c8475ba8ab8
#owner: https://api.github.com/users/joaopfsilva

#!/bin/bash

# 1. `./main.sh staging`            -> enter staging ssh
# 2. `./main.sh production`         -> enter production ssh
# 3. `./main.sh staging disable`    -> disable staging debug
# 4. `./main.sh production disable` -> disable staging debug

# get the service name and optional disable argument from command-line arguments
service_name=$1
disable=$2

# get the last version of the service using gcloud command
versions=$(gcloud app versions list --service=$service_name --sort-by '~version.id' --format 'value(version.id)')
last_version=$(echo "$versions" | head -n 1 | xargs)

# get the instance ID
instances=$(gcloud app instances list --service=$service_name --version=$last_version --format 'value(id)')
instance_id=$(echo "$instances" | head -n 1 | xargs)

echo $instances
echo $service_name
echo $instance_id
echo $last_version

# run gcloud app instances ssh with the last version and instance ID
if [ "$disable" == "disable" ]; then
  echo "Disable debug"
  gcloud app instances disable-debug $instance_id
else
  echo "Enter SSH"
  gcloud app instances ssh --service=$service_name $instance_id --version=$last_version --container=gaeapp
fi


