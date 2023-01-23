#date: 2023-01-23T16:52:35Z
#url: https://api.github.com/gists/c3ffaac0ce6ef8003903e0fd689124a2
#owner: https://api.github.com/users/AwdotiaRomanowna

echo "Port 47622" >> /etc/ssh/sshd_config
systemctl restart sshd
touch /tmp/"$1"