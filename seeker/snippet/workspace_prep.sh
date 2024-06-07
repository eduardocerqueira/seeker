#date: 2024-06-07T17:05:28Z
#url: https://api.github.com/gists/1c8f722838dbdfadef65e8cfcdbb06c8
#owner: https://api.github.com/users/fkdraeb

#!/bin/bash

# Update package lists
sudo apt update

######################################################################################

echo "Installing Docker ..."
sudo apt install -y docker.io

echo "Installing Docker Compose ..."
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

set +e

echo "... Giving docker management as a non-root user"

# Function to add a group if it does not already exist
add_group_if_not_exists() {
  local group_name="$1"
  if ! getent group "$group_name" > /dev/null 2>&1; then
    sudo groupadd "$group_name"
    echo "Group '$group_name' created."
  else
    echo "Group '$group_name' already exists."
  fi
}

add_group_if_not_exists "docker"

# Re-enable exit on non-zero exit status
set -e

sudo usermod -aG docker $USER

######################################################################################

echo "Installing JDK 21 ..."

# Removing previous installed versions if they exist
if update-alternatives --list java; then
  sudo update-alternatives --remove-all java
fi
if update-alternatives --list javac; then
  sudo update-alternatives --remove-all javac
fi
sudo rm -rf /opt/java/*

sudo wget -O "/tmp/jdk-21.tar.gz" "https://download.java.net/java/GA/jdk21.0.2/f2283984656d49d69e91c558476027ac/13/GPL/openjdk-21.0.2_linux-x64_bin.tar.gz"
sudo mkdir -p "/opt/java"
sudo tar -xf "/tmp/jdk-21.tar.gz" -C "/opt/java"
sudo rm "/tmp/jdk-21.tar.gz"

# Setting Environment variables
JAVA_HOME=/opt/java/jdk-21.0.2
echo "export JAVA_HOME=$JAVA_HOME" | sudo tee -a /etc/profile
echo "export PATH=\$JAVA_HOME/bin:\$PATH" | sudo tee -a /etc/profile
sudo update-alternatives --install "/usr/bin/java" "java" "$JAVA_HOME/bin/java" 1
sudo update-alternatives --install "/usr/bin/javac" "javac" "$JAVA_HOME/bin/javac" 1

source /etc/profile

######################################################################################

echo "Installing Maven ..."

sudo apt install maven

######################################################################################

echo "Installing k8s ..."

curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

######################################################################################

echo "Installing Minikube ..."

curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube && rm minikube-linux-amd64

######################################################################################

# Function to display version information
check_version() {
  echo "----------------------------------------"
  echo "> $1 Version                            "
  echo "----------------------------------------"
  $2
  echo ""
}

print_not_installed() {
  echo "❌ $1 is not installed."
}

# Check if commands are available and display their versions
echo "========================================"
echo "|         🚀 Installation Check        |"
echo "========================================"

if command -v java &> /dev/null; then
  check_version "☕ Java" "java -version 2>&1 | head -n 1"
else
  print_not_installed "Java"
fi

if command -v docker &> /dev/null; then
  check_version "🐳 Docker" "docker --version"
else
  print_not_installed "Docker"
fi

if command -v docker &> /dev/null; then
  check_version "🐳 Docker-Compose" "docker-compose --version"
else
  print_not_installed "Docker-Compose"
fi

if command -v mvn &> /dev/null; then
  check_version "🔨 Maven" "mvn -v"
else
  print_not_installed "Maven"
fi

if command -v kubectl &> /dev/null; then
  check_version "🏗️ Kubernetes " "kubectl version --client"
else
  print_not_installed "Kubernetes"
fi

if command -v minikube &> /dev/null; then
  check_version "🖥️ Minikube " "minikube version"
else
  print_not_installed "Minikube"
fi

echo "========================================"
echo "|        ✅ Check Completed            |"
echo "========================================"
