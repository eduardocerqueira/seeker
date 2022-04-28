#date: 2022-04-28T17:09:30Z
#url: https://api.github.com/gists/594d7147cb61a04b38caf96d70caf571
#owner: https://api.github.com/users/bartprokop

pacman -S hdparm
pacman -S smartmontools

smartctl -i /dev/sdb
# smartctl 7.3 2022-02-28 r5338 [x86_64-linux-5.17.4-arch1-1] (local build)
# Copyright (C) 2002-22, Bruce Allen, Christian Franke, www.smartmontools.org
# 
# === START OF INFORMATION SECTION ===
# Model Family:     Western Digital Ultrastar He10/12
# Device Model:     WDC WD80EMAZ-00WJTA0
# Serial Number:    2SG8YU8J
# LU WWN Device Id: 5 000cca 27dc412e4
# Firmware Version: 83.H0A83
# User Capacity:    8,001,563,222,016 bytes [8.00 TB]
# Sector Sizes:     512 bytes logical, 4096 bytes physical
# Rotation Rate:    5400 rpm
# Form Factor:      3.5 inches
# Device is:        In smartctl database 7.3/5319
# ATA Version is:   ACS-2, ATA8-ACS T13/1699-D revision 4
# SATA Version is:  SATA 3.2, 6.0 Gb/s (current: 6.0 Gb/s)
# Local Time is:    Thu Apr 28 17:57:18 2022 BST
# SMART support is: Available - device has SMART capability.
# SMART support is: Enabled

# inspect NTFS drive, before zeroing it
mount -t ntfs3 /dev/sdb1 /mnt
find /mnt

