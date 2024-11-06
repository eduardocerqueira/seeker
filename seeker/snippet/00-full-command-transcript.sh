#date: 2024-11-06T17:07:12Z
#url: https://api.github.com/gists/eac5f7d06fae2b12a2d2dd02d896ded2
#owner: https://api.github.com/users/rafaelcalleja

#!/bin/bash

# do this on localhost (deployment host)
# ensure that there's a local ssh private key
ssh-keygen -t rsa -N '' -f ~/.ssh/id_rsa

# now make sure that the public key is in the second host's authorized_keys
# then do a test ssh connection to make sure it works, and to add the host
# to known hosts

apt-get update && \
  apt-get purge -y nano && \
  apt-get install -y git vim tmux fail2ban build-essential python2.7 python-dev libssl-dev libffi-dev lxc lxc-dev

curl --silent --show-error --retry 5 https://bootstrap.pypa.io/get-pip.py | sudo python2.7

pip install -U ansible==2.2.0 lxc-python2

git config --global user.email "user@example.com"
git config --global user.name "User"
git config --global push.default matching
git config --global --add gitreview.username "user"

mkdir -p ~/.ansible
git clone https://github.com/openstack/openstack-ansible-plugins.git ~/.ansible/plugins
cd ~/.ansible/plugins
git fetch https://git.openstack.org/openstack/openstack-ansible-plugins refs/changes/38/400338/6 && git cherry-pick FETCH_HEAD

cd ~
ansible-playbook -i 01-inventory.ini 02-playbook.yml -vvv