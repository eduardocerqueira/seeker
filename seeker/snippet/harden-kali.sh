#date: 2023-12-08T17:03:50Z
#url: https://api.github.com/gists/0295c730a81f4dbabd7e5dcb01dd37c3
#owner: https://api.github.com/users/amigus

#!/bin/sh

UMASK=027

sed -E -idistro /etc/login.defs \
    -e "s/^(UMASK\s+).*$/\1${UMASK}/" \
    -e 's/^(DEFAULT_HOME\s+).*$/\1no/' \
    
apt -y update
apt -y full-upgrade
apt -y autoremove
apt -y install sshguard

systemctl enable sshguard
systemctl start sshguard

echo 'PermitRootLogin no' >| /etc/ssh/sshd_config.d/10-no-permit-root-login.conf
echo "session optional\t\t\t pam_umask.so umask=${UMASK}" >> /etc/pam.d/common-session