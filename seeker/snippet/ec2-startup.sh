#date: 2023-08-01T17:03:48Z
#url: https://api.github.com/gists/f135957f0ec8d62587c087e5a7e61c34
#owner: https://api.github.com/users/cheseremtitus24

#!/bin/sh
export PATH=/usr/local/bin:$PATH;

yum update
yum install docker -y
service docker start
# Docker login notes:
#   - For no email, just put one blank space.
#   - Also the private repo protocol and version are needed for docker
#     to properly setup the .dockercfg file to work with compose
docker login --username="someuser" --password="asdfasdf" --email=" " https: "**********"
mv /root/.dockercfg /home/ec2-user/.dockercfg
chown ec2-user:ec2-user /home/ec2-user/.dockercfg
usermod -a -G docker ec2-user
curl -L https://github.com/docker/compose/releases/download/1.7.1/docker-compose-`uname -s`-`uname -m` > /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
chown root:docker /usr/local/bin/docker-compose
cat <<EOF >/home/ec2-user/docker-compose.yml
nginx:
  image: nginx
  ports:
    - "80:80"
EOF
chown ec2-user:ec2-user /home/ec2-user/docker-compose.yml
/usr/local/bin/docker-compose -f /home/ec2-user/docker-compose.yml up -d
 -d
