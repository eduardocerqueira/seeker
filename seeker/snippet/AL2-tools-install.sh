#date: 2023-03-20T16:57:59Z
#url: https://api.github.com/gists/d1416de2e4df38e9b615de51ebd12035
#owner: https://api.github.com/users/chrissaddoris

#!/bin/bash
PATH=/bin:/usr/bin
set -x

mkdir ~/tmp
cd ~/tmp

sudo amazon-linux-extras install epel -y
sudo yum update -y

## aws cli2
sudo yum remove aws-cli -y
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
 . ~/.bashrc

## kubectl
curl -O https://s3.us-west-2.amazonaws.com/amazon-eks/1.25.6/2023-01-30/bin/linux/amd64/kubectl
chmod +x ./kubectl
sudo mv ./kubectl /usr/local/bin

## psql
sudo tee /etc/yum.repos.d/pgdg.repo<<EOF
[pgdg14]
name=PostgreSQL 14 for RHEL/CentOS 7 - x86_64
baseurl=https://download.postgresql.org/pub/repos/yum/14/redhat/rhel-7-x86_64
enabled=1
gpgcheck=0
EOF
sudo yum install postgresql14 -y

## devspace
curl -L -o devspace "https://github.com/loft-sh/devspace/releases/latest/download/devspace-linux-amd64" && sudo install -c -m 0755 devspace /usr/local/bin

## anaconda
curl https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh -o ~/tmp/anaconda.sh
bash ~/tmp/anaconda.sh -b -p $HOME/anaconda