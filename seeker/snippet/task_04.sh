#date: 2022-06-24T17:09:56Z
#url: https://api.github.com/gists/10b8e252534a6660036e8ade9f69ff8b
#owner: https://api.github.com/users/syedali3762

#!/bin/bash

no_root=100
ARCH=$(uname -i)

if [[ "$UID" -eq "0" ]]
then


which aws &>/dev/null || {
       echo" Install aws-cli"
    which curl&> /dev/null || apt install curl -y
    which unzip &> /dev/null || apt install unzip -y

    curl "https://awscli.amazonaws.com/awscli-exe-linux-$ARCH.zip" -o "awscliv2"

    unzip $PWD/awscliv2.zip
     } && {
      AWS_ACCESS_KEY_ID=$(aws --region=us-east-1 ssm get-parameter --name "MY_ACCESS_KEY" --with-decryption --output text --query Parameter.Value)
      echo ${AWS_ACCESS_KEY_ID}

      AWS_SECRET_ACCESS_KEY=$(aws --region=us-east-1 ssm get-parameter --name "MY_SECRET_KEY" --with-decryption --output text --query Parameter.Value)
      echo ${AWS_SECRET_ACCESS_KEY}
}
else
      echo $not_sudo
fi


