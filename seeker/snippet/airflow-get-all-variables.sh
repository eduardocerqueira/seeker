#date: 2021-10-28T16:59:45Z
#url: https://api.github.com/gists/0b5bf57e0aac31b85a6636af480ec904
#owner: https://api.github.com/users/xlyk

#!/bin/bash

for KEY in $(airflow variables list  | tail -n +2); do
  echo "$KEY=$(airflow variables get $KEY)"
done