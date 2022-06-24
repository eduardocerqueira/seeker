#date: 2022-06-24T17:07:06Z
#url: https://api.github.com/gists/5fe47511b58a378653fd09ce80bb0a5e
#owner: https://api.github.com/users/syedali3762

#!/bin/bash

no_root=100
ARCH=$(uname -i)

if [[ "$UID" -ne "0" ]]
then
   echo " Must be root"
   exit $no_root

fi

which aws &>/dev/null || {
        echo" Install aws-cli"
    which curl&> /dev/null || apt install curl -y
    which unzip &> /dev/null || apt install unzip -y

    curl "https://awscli.amazonaws.com/awscli-exe-linux-$ARCH.zip" -o "awscliv>

    unzip $PWD/awscliv2.zip
   unzip $PWD/awscliv2.zip
    sudo $PWD/aws/install

       } && {
       AWS_VERSION=$(aws --version |awk '{print $1}'  | tr '/' '-')
                echo "$AWS_VERSION is already installed"
       }

