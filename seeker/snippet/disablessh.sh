#date: 2025-09-16T16:55:39Z
#url: https://api.github.com/gists/304988ff843e1f5eb58f9f57f3410df6
#owner: https://api.github.com/users/dinhkarate

sed -i -E 's/#?PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl restart sshd || systemctl restart ssh