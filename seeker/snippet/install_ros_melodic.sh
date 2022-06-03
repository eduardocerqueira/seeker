#date: 2022-06-03T17:07:06Z
#url: https://api.github.com/gists/ca1ede112e27f8e14bc28a6ab435224e
#owner: https://api.github.com/users/chris24sahadeo

#!/bin/bash

# sudo wget -qO- https://gist.githubusercontent.com/chris24sahadeo/ca1ede112e27f8e14bc28a6ab435224e/raw/03bfee3bf3679172bb6e047197fb97b0f64d33b7/install_ros_melodic.sh | sudo bash

sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt install curl # if you haven't already installed curl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt -y update 
sudo apt -y install ros-melodic-desktop-full 
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc
sudo apt -y install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential
sudo apt -y install python-rosdep
sudo rosdep init
rosdep update
