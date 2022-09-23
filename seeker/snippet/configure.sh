#date: 2022-09-23T17:20:39Z
#url: https://api.github.com/gists/7edcc8a1f6ef0b9af4b3f0e55c50cb95
#owner: https://api.github.com/users/sprig

#!/bin/sh

setup-timezone -z UTC

cat <<-EOF > /etc/network/interfaces
iface lo inet loopback
iface eth0 inet dhcp
EOF

ln -s networking /etc/init.d/net.lo
ln -s networking /etc/init.d/net.eth0

sed -Ei \
	-e 's/^[# ](rc_depend_strict)=.*/\1=NO/' \
	-e 's/^[# ](rc_logger)=.*/\1=YES/' \
	-e 's/^[# ](unicode)=.*/\1=YES/' \
	/etc/rc.conf

rc-update add acpid default
rc-update add chronyd default
rc-update add crond default
rc-update add sshd default
rc-update add net.eth0 default
rc-update add net.lo boot
rc-update add termencoding boot
rc-update add haveged boot

cat <<EOF > /etc/motd
Welcome to Alpine!
EOF

mkdir -m 700 -p /root/.ssh
curl https://github.com/nrocco.keys | tee /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

chsh --shell /bin/bash root

cat <<'EOF' > /root/.bash_profile
PS1="[\u@\h:\w${_p}] "
export EDITOR=/usr/bin/vim
EOF