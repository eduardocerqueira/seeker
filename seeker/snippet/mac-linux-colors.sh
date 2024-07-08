#date: 2024-07-08T16:54:47Z
#url: https://api.github.com/gists/6348692ffd9903d249907fb5212056d5
#owner: https://api.github.com/users/MattCurryCom

#!/bin/bash
#######################
#### Author: Matt Curry
#### Date 07/08/24
#### Adds Colors to mac shell, so you can see the difference between directories/files/etc...
#######################

# or for .profile
echo " " >> ~/.profile
echo "# Mac Color Fix" >> ~/.profile
echo "export CLICOLOR=1" >> ~/.profile
echo "export LSCOLORS=GxFxCxDxBxegedabagaced" >> ~/.profile

# Re-Open Terminal after applying.