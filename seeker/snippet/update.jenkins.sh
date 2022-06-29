#date: 2022-06-29T17:07:36Z
#url: https://api.github.com/gists/cb628c1359b468fe1a696bd27e9b4ec0
#owner: https://api.github.com/users/marceloneias

# Download last version of Jenkins (replace <new-version-number>)
cd /home/ubuntu/downloads
wget http://updates.jenkins-ci.org/download/war/<new-version-number>/jenkins.war

# Create backup of current version (replace <old-version-number>)
cp /usr/share/java/jenkins.war /home/ubuntu/jenkins-versions/jenkins-<old-version-number>.war

# Copy new version (replace <new-version-number>)
cp /home/ubuntu/donwload/jenkins-<new-version-number>.war /usr/share/java/jenkins.war

# Stop and start jenkins
sudo systemctl stop jenkins

sudo systemctl start jenkins