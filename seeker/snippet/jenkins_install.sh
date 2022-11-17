#date: 2022-11-17T17:04:07Z
#url: https://api.github.com/gists/732325265c4119fdeac95984cb8e5f04
#owner: https://api.github.com/users/MrWalshyType2

#!/bin/bash
# Setup
echo "Configuring system dependent dependencies"
if type apt > /dev/null; then
  echo "Using apt"
  pkg_mgr=apt
  java_deps="openjdk-17-jdk openjdk-17-jre"
elif type yum > /dev/null; then
  echo "Using yum"
  pkg_mgr=yum
  java_deps=java-17-openjdk-devel
fi

echo "Updating dependencies"
sudo ${pkg_mgr} update -y

echo "Installing dependencies: java, wget, git" 
sudo ${pkg_mgr} remove -y java 
sudo ${pkg_mgr} install -y ${java_deps} wget git

# Configure Jenkins user
echo "Configuring jenkins user"
sudo useradd -m -s /bin/bash jenkins

# Download Jenkins WAR file
echo "Downloading latest jenkins WAR"
sudo su - jenkins -c "curl -L https://updates.jenkins-ci.org/latest/jenkins.war --output jenkins.war"

# Create Jenkins service file
echo "Setting up jenkins service"
# Redirect normal stdout to bit bucket
sudo tee /etc/systemd/system/jenkins.service << EOF > /dev/null
[Unit]
Description=Jenkins Server

[Service]
User=jenkins
WorkingDirectory=/home/jenkins
ExecStart=/usr/bin/java -jar /home/jenkins/jenkins.war

[Install]
WantedBy=multi-user.target
EOF

# Reload service files and start jenkins
sudo systemctl daemon-reload
sudo systemctl enable jenkins
sudo systemctl restart jenkins

# Become Jenkins user and wait for password to become available
sudo su - jenkins << EOF
until [ -f .jenkins/secrets/initialAdminPassword ]; do
    sleep 1
    echo "waiting for initial admin password"
done
until [[ -n "\$(cat  .jenkins/secrets/initialAdminPassword)" ]]; do
    sleep 1
    echo "waiting for initial admin password"
done
echo "initial admin password: "**********"
EOFsecrets/initialAdminPassword)"
EOF