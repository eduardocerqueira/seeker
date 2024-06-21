#date: 2024-06-21T17:07:50Z
#url: https://api.github.com/gists/cc4babaf28cdbbbc9f7af5e7ec57f94f
#owner: https://api.github.com/users/thimslugga

#!/bin/bash

# Usage: bash <(curl -sL https://gist.github.com/thimslugga/<>/sanitze-clean.sh)
#
# https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/8/html-single/configuring_and_managing_cloud-init_for_rhel_8/index

function info() {
  echo -e "\e[32m${1}\e[0m"
}

info 'Clean Yum'
yum clean all

info 'Remove SSH keys'
[ -f /home/ec2-user/.ssh/authorized_keys ] && rm -f /home/ec2-user/.ssh/authorized_keys

info  'Cleanup log files'
find /var/log -type f \
  | while read f; do 
    echo -ne '' > $f; 
  done

info 'Cleanup bash history'
unset HISTFILE
[ -f /root/.bash_history ] && rm -f /root/.bash_history
[ -f /home/ec2-user/.bash_history ] && rm -f /home/ec2-user/.bash_history

info 'The instance has been sanitized!'