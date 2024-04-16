#date: 2024-04-16T16:45:55Z
#url: https://api.github.com/gists/aef0968829883110e24b107f7278255f
#owner: https://api.github.com/users/codingforentrepreneurs

#!/bin/bash
# Create user "tars"
sudo useradd -r -s /sbin/nologin tars
sudo usermod -aG sudo tars
echo "tars ALL=(ALL) NOPASSWD:ALL" | sudo tee /etc/sudoers.d/tars

# Define all required directories
directories=(
    /data/my-config
    /var/log/zookeeper
    /var/log/kafka
    /opt/kafka
    /tmp/zookeeper
    /data/zookeeper
    /data/kafka
)

# Loop through each directory
for dir in "${directories[@]}"; do
    # Create the directory with sudo, avoiding errors if it already exists
    sudo mkdir -p "$dir"
    
    # Change the ownership to 'tars' user and group, recursively
    sudo chown -R tars:tars "$dir"
done

# Install Java and Required packages
sudo apt-get update && sudo apt-get -y install wget ca-certificates zip net-tools vim nano tar netcat openjdk-8-jdk

# Add file limits configs - allow to open 100,000 file descriptors
echo "* hard nofile 100000* soft nofile 100000" | sudo tee --append /etc/security/limits.conf

# update memory swap
sudo sysctl vm.swappiness=1
echo 'vm.swappiness=1' | sudo tee --append /etc/sysctl.conf

# Download Kafka (including Zookeeper) from
# https://kafka.apache.org/downloads
curl https://dlcdn.apache.org/kafka/3.7.0/kafka_2.13-3.7.0.tgz -o kafka.tgz
tar -xvzf kafka.tgz
mv kafka_*/* /opt/kafka/
rm kafka.tgz