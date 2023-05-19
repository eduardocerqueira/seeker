#date: 2023-05-19T16:43:52Z
#url: https://api.github.com/gists/271f4854d6b96915f843121f7af28334
#owner: https://api.github.com/users/ycabrer

#!/bin/bash

aws ecr describe-repositories | jq '.repositories[].repositoryName' | xargs -I {} aws ecr put-lifecycle-policy --repository-name {} --lifecycle-policy-text "file://policy.json"
