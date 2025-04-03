#date: 2025-04-03T17:08:23Z
#url: https://api.github.com/gists/0f416b0a4abe683000342310fe6bcdab
#owner: https://api.github.com/users/sadam-is-sleeping

#!/bin/bash

# Bash script to replace Jenkins' IP Address
JENKINS_CONFIG_FILE="/var/lib/jenkins/jenkins.model.JenkinsLocationConfiguration.xml"
# Get TOKEN from Amazon, with short lifetime (10s)
TOKEN=`curl -X PUT "http: "**********": 10"`
# Get IP Address
IPADDR=`curl -H "X-aws-ec2-metadata-token: "**********"://169.254.169.254/latest/meta-data/public-ipv4`
echo "Current IP address (IPv4): $IPADDR"
# Replace jenkinsUrl
sudo sed -i "s|<jenkinsUrl>.*</jenkinsUrl>|<jenkinsUrl>${IPADDR}</jenkinsUrl>|g" "$JENKINS_CONFIG_FILE"
# Restart Jenkins
sudo systemctl restart jenkinsS_CONFIG_FILE"
# Restart Jenkins
sudo systemctl restart jenkins