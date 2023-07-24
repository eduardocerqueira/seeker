#date: 2023-07-24T16:38:42Z
#url: https://api.github.com/gists/a480cd039e7c31eea873360622e0b61e
#owner: https://api.github.com/users/SYNchroACK

#!/bin/bash
instance_id="$1"
region="$2"
profile="$3"
echo "Instance: "$instance_id
echo "Region: "$region
echo "Profile: "$profile
aws ssm start-session --target $instance_id --profile $profile --region $region