#date: 2023-03-13T16:52:06Z
#url: https://api.github.com/gists/51104b767a1228bc17177363d80858bf
#owner: https://api.github.com/users/PavaniVitor

#!/bin/bash

# example how to export ROS 1 master

## ros master terminal ##
# change enp4s0 to your network device
ROS_IP=$(ip -4 addr show enp4s0 | grep -oP --color=never '(?<=inet\s)\d+(\.\d+){3}')

export ROS_MASTER_URI=http://$ROS_IP:11311
echo $ROS_MASTER_URI

# run roscore or roslaunch as usual
roscore

## ros client node terminal ##

# paste ROS_MASTER_URI from master terminal
export ROS_MASTER_URI=http://'MASTER_IP':11311

# change wlo0 to your network device
export ROS_IP=$(ip -4 addr show wlo0 | grep -oP --color=never '(?<=inet\s)\d+(\.\d+){3}')

# run ros node or roslaunch as usual
rosrun rqt_image_view rqt_image_view
