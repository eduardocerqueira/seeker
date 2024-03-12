#date: 2024-03-12T17:03:14Z
#url: https://api.github.com/gists/250303d38ff23e413b8a6509d8927ea2
#owner: https://api.github.com/users/heittpr

#!/bin/sh

sudo apt --assume-yes install slapd ldap-utils sssd
sudo dpkg-reconfigure slapd

cat << EOF >> /etc/sssd/sssd.conf
[sssd]
config_file_version = 2
domains = dcc.ufmg.br

[domain/dcc.ufmg.br]
id_provider = ldap
auth_provider = ldap
ldap_uri = ldap://ldap.dcc.ufmg.br
cache_credentials = True
ldap_search_base = dc=dcc,dc=ufmg,dc=br
override_homedir=/home/%u
EOF

sudo chown root /etc/sssd/sssd.conf
sudo chmod 0600 /etc/sssd/sssd.conf
sudo pam-auth-update --enable mkhomedir
sudo systemctl restart slapd
sudo systemctl restart sssd