#date: 2022-01-21T16:47:50Z
#url: https://api.github.com/gists/1879fcb91fe42b101007d7c6e03b7442
#owner: https://api.github.com/users/pkla

#!/bin/bash

sudo yum install -y R

INSTALL_FILE=rstudio-server-rhel-2021.09.2-382-x86_64.rpm
wget https://download2.rstudio.org/server/centos7/x86_64/$INSTALL_FILE
sudo yum install -y $INSTALL_FILE
rm $INSTALL_FILE