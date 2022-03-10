#date: 2022-03-10T17:02:55Z
#url: https://api.github.com/gists/46413a88d111dfa56aef5890d5526fdc
#owner: https://api.github.com/users/s5x

ssh-keygen -t dsa -f id_dsa -m PEM -C "`whoami`@`hostname -s` key (`date +'%Y-%m-%d'`)" -b 1024
ssh-keygen -t rsa -f id_rsa -m PEM -C "`whoami`@`hostname -s` key (`date +'%Y-%m-%d'`)" -b 2048
ssh-keygen -t ed25519 -f id_ed25519 -C "`whoami`@`hostname -s` key (`date +'%Y-%m-%d'`)" -o -a 256