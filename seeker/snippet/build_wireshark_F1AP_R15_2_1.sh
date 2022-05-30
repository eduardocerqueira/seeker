#date: 2022-05-30T17:04:59Z
#url: https://api.github.com/gists/ca6583f344ac7ac70aec09aa252faa1a
#owner: https://api.github.com/users/caio-vinicius

#!/bin/sh
# This shell script is made by SyneArt <sa@syneart.com>
#######################################
# BUILD WIRESHARK ON UBUNTU OR DEBIAN #
#######################################

# |          THIS SCRIPT IS TESTED CORRECTLY ON          |
# |------------------------------------------------------|
# | OS             | Wireshark      | Test | Last test   |
# |----------------|----------------|------|-------------|
# | Ubuntu 20.04.1 | Commit:a679ae6 | OK   | 05 Nov 2020 |
# | Ubuntu 18.04.1 | Commit:a679ae6 | OK   | 20 Nov 2018 |
# | Ubuntu 16.04.5 | Commit:a679ae6 | OK   | 20 Nov 2018 |
# | Ubuntu 14.04.5 | Commit:a679ae6 | OK   | 20 Nov 2018 |
# | Debian 9.6     | Commit:a679ae6 | OK   | 20 Nov 2018 |

# 1. KEEP UBUNTU OR DEBIAN UP TO DATE
sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y autoremove

# 2. INSTALL THE DEPENDENCIES
sudo apt-get install -y build-essential git cmake 

# CMAKE3
sudo apt-get install -y cmake3

# GUI
sudo apt-get install -y qttools5-dev qttools5-dev-tools libqt5svg5-dev qtmultimedia5-dev

# PCAP
sudo apt-get install -y libpcap-dev

# Dev file (On Ubuntu 20.04)
sudo apt-get install -y libc-ares-dev

# CRYPT
sudo apt-get install -y libgcrypt20-dev

# GLIB2
sudo apt-get install -y libglib2.0-dev

# LEX & YACC
sudo apt-get install -y flex bison

# 3. BUILD THE WIRESHARK
git clone https://github.com/wireshark/wireshark ~/wireshark_f1ap_r15_2_1
cd ~/wireshark_f1ap_r15_2_1
git checkout a679ae6 # F1AP R15.2.1
mkdir build
cd build
cmake -DDISABLE_WERROR=true ../ # g++
make -j`nproc` && {
  echo "\nBuild Success!"
  echo "You can execute the Wireshark by command \"sudo ./wireshark\""
  echo "at \"`pwd`/run\""
}
