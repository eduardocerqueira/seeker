#date: 2023-04-19T17:05:22Z
#url: https://api.github.com/gists/b143c7fe02e9eb3753ee06d60bb2c869
#owner: https://api.github.com/users/blewis-grax

#!/bin/bash

 Check if Python 3 is installed
if ! command -v python3 &> /dev/null
then
    echo "Python 3 is not installed. Installing Python 3..."
    # Install Python 3 using the official Python installer
    curl https://www.python.org/ftp/python/3.10.0/python-3.10.0-macosx11.0.pkg -o python-3.10.0-macosx11.0.pkg
    sudo installer -pkg python-3.10.0-macosx11.0.pkg -target /
    rm python-3.10.0-macosx11.0.pkg
else
    echo "Python 3 is already installed. Skipping installation."
fi

# Add Python 3 to the system PATH
echo 'export PATH="/Library/Frameworks/Python.framework/Versions/3.10/bin:$PATH"' | sudo tee -a /etc/paths > /dev/null
source /etc/paths

# Install Apple Command Line Developer Tools
# xcode-select --install
echo "Checking for the existence of the Apple Command Line Developer Tools"
/usr/bin/xcode-select -p &> /dev/null
if [[ $? -ne 0 ]]; then
    echo "Apple Command Line Developer Tools not found."
    touch /tmp/.com.apple.dt.CommandLineTools.installondemand.in-progress;
    installationPKG=$(/usr/sbin/softwareupdate --list | /usr/bin/grep -B 1 -E 'Command Line Tools' | /usr/bin/tail -2 | /usr/bin/awk -F'*' '/^ \/ {print $2}' | /usr/bin/sed -e 's/^ *Label: //' -e 's/^ *//' | /usr/bin/tr -d '\n')
    echo "Installing ${installationPKG}"
    /usr/sbin/softwareupdate --install "${installationPKG}" --verbose
else
    echo "Apple Command Line Developer Tools are already installed."
fi

#add python3 to $PATH and import SystemDonfiguration module
set -eo pipefail

if ! [ -x /usr/bin/python3 ] || ! [ -x /usr/bin/pip3 ]; then
    echo "/usr/bin/python3 or /usr/bin/pip3 not found" >&2
    exit 1
fi

if [ "$(id -u)" != 0 ]; then
    echo "Must be run as root" >&2
    exit 1
fi

export PATH="/usr/bin:$PATH" # make sure /usr/bin/python3 preferred

pip3 install pyobjc-framework-SystemConfiguration
python3 -c 'import SystemConfiguration'