#date: 2023-01-10T16:51:09Z
#url: https://api.github.com/gists/ce5ff228e3c3376b7485ebbada0cdbb1
#owner: https://api.github.com/users/dartasensi

#!/bin/bash
defaultUser="dartasensi"
defaultEmail="dartasensi@users.noreply.github.com"

rsaKeyFile=/home/cloudshell-user/.ssh/id_rsa
if [[ ! -f "$rsaKeyFile" ]] ; then
    #add rsa key
    ssh-keygen -b 2048 -t rsa -f "$rsaKeyFile" -q -N ""
    echo "Please copy the following into your GitHub profile here: https://github.com/settings/ssh/new
    "
    cat /home/cloudshell-user/.ssh/id_rsa.pub

    read -r -p "Press any key to continue...
    "
fi

read -p "GitHub Username: [$defaultUser]" uservar
read -p "GitHub Email: [$defaultEmail]" emailvar
uservar=${uservar:-${defaultUser}}
emailvar=${emailvar:-${defaultEmail}}

echo "Using... 
username: $uservar
email: $emailvar
"

git config --global user.name $uservar
git config --global user.email $emailvar
git config --global credential.helper cache

sshConfig=<<EOD
Host github.com
    User git
    Hostname github.com
    PreferredAuthentications publickey
    IdentityFile /home/cloudshell-user/.ssh/id_rsa
EOD

echo $sshConfig >> /home/cloudshell-user/.ssh/config
chmod 600 /home/cloudshell-user/.ssh/config

eval $(ssh-agent -s)
ssh-add /home/cloudshell-user/.ssh/id_rsa

sudo ssh-keyscan -H github.com >> ~/.ssh/known_hosts

# DART evaluate this repo: https://github.com/aws-samples/extensions-for-aws-managed-shells.git
##get cloud extensions script
#git clone git@github.com:ImIOImI/extensions-for-aws-managed-shells.git

##run cloud extensions script
#cd extensions-for-aws-managed-shells
#sh extensions-for-aws-managed-shells.sh
cd ~

##add all the eks clusters you have access to
#clusters=$(aws eks list-clusters | grep -oP '"(.*?)"' | sed 's/"clusters"//' | sed -e 's/^"//' -e 's/"$//')
#echo $clusters
#for val in $clusters ; do
#    aws eks update-kubeconfig --name $val --alias $val
#done