#date: 2023-05-23T17:08:38Z
#url: https://api.github.com/gists/abfb81bb90f4c83ca79e920a0bf67c8a
#owner: https://api.github.com/users/Dangercoder

#!/bin/bash

# Download the Microsoft package
wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb

# Install the package
sudo dpkg -i packages-microsoft-prod.deb

# Remove the package
rm packages-microsoft-prod.deb

# Update the packages and install .NET SDK
sudo apt-get update
sudo apt-get install -y dotnet-sdk-7.0

# Update the packages and install ASP.NET Core Runtime
sudo apt-get update
sudo apt-get install -y aspnetcore-runtime-7.0

# Install .NET Runtime
sudo apt-get install -y dotnet-runtime-7.0

# Install Clojure Main tool
dotnet tool install --global --version 1.12.0-alpha7 Clojure.Main

# Add the tool to the PATH
echo 'export PATH="$PATH:$HOME/.dotnet/tools"' >> ~/.bashrc
source ~/.bashrc