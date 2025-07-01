#date: 2025-07-01T16:51:03Z
#url: https://api.github.com/gists/ac6604d809899bfabacf7c2275fc54e1
#owner: https://api.github.com/users/scr34m

wget https://github.com/torvalds/linux/raw/master/drivers/hwmon/lm75.h
wget https://github.com/torvalds/linux/raw/master/drivers/hwmon/nct6775.h
wget https://github.com/torvalds/linux/raw/master/drivers/hwmon/nct6775-core.c
wget https://github.com/torvalds/linux/raw/master/drivers/hwmon/nct6775-platform.c
wget https://github.com/torvalds/linux/raw/master/drivers/hwmon/nct6775-i2c.c

# tell DKMS where this source stuff it
sudo ln -s `realpath .` /usr/src/nct6775-wim

# compile files above
sudo dkms install nct6775/wim
