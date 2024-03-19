#date: 2024-03-19T16:47:51Z
#url: https://api.github.com/gists/35f17820b56141a0f21afbf596e000f0
#owner: https://api.github.com/users/kernelzeroday

#needs root access, remove the backdoored curve hostkeys that infect everything now and revert to safe rsa
rm -v /etc/ssh/*ed25519*
sed -i 's/#HostKey\ \/etc\/ssh\/ssh_host_rsa_key/HostKey\ \/etc\/ssh\/ssh_host_rsa_key/g' /etc/ssh/sshd_config
systemctl restart ssh