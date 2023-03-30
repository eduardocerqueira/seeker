#date: 2023-03-30T17:05:06Z
#url: https://api.github.com/gists/365bde107c88af615456a81914a05300
#owner: https://api.github.com/users/SuperPhenotype

#!/usr/bin/env bash
# 
#    Author: Markus (MawKKe) ekkwam@gmail.com
#    Date: 2018-03-19
#
#
# What?
#
#    Linux dm-crypt + dm-integrity + dm-raid (RAID1) 
#
#    = Secure, redundant array with data integrity protection
#
# Why? 
#
#   You see, RAID1 is dead simple tool for disk redundancy, 
#   but it does NOT protect you from bit rot. There is no way
#   for RAID1 to distinguish which drive has the correct data if rot occurs.
#   This is a silent killer.
#
#   But with dm-integrity, you can now have error detection
#   at the block level. But it alone does not provide error correction
#   and is pretty useless with just one disk (disks fail, shit happens).
#
#   But if you use dm-integrity *below* RAID1, now you have disk redundancy,
#   AND error checking AND error correction. Invalid data received from
#   a drive will cause a checksum error which the RAID array notices and 
#   replaces with correct data.
#
#   If you throw encryption into the mix, you'll have secure,
#   redundant array. Oh, and the data integrity can be protected with 
#   authenticated encryption, so no-one can tamper your data maliciously. 
#   
#   How cool is that?
#
#   Also: If you use RAID1 arrays as LVM physical volumes, the overall 
#   architecture is quite similar to ZFS! All with native Linux tools, 
#   and no hacky solaris compatibility layers or licencing issues!
#
#   (I guess you can use whatever RAID level you want, but RAID1 is the 
#    simplest and fastest to set up)
# 
#
#   Let's try it out!
#
#   ---
#   NOTE: The dm-integrity target is available since Linux kernel version 4.12.
#   NOTE: This example requires LUKS2 which is only recently released (2018-03)
#   NOTE: The authenticated encryption is still experimental (2018-03)
#   ---

set -eux

# 1) Make dummy disks

cd /tmp

truncate -s 500M disk1.img
truncate -s 500M disk2.img

# Format the disk with luksFormat:

dd if=/dev/urandom of=key.bin bs=512 count=1

cryptsetup luksFormat -q --type luks2 --integrity hmac-sha256 disk1.img key.bin
cryptsetup luksFormat -q --type luks2 --integrity hmac-sha256 disk2.img key.bin

# The luksFormat's might take a while since the --integrity causes the disks to be wiped.
# dm-integrity is usually configured with 'integritysetup' (see below), but as 
# it happens, cryptsetup can do all the integrity configuration automatically if
# the --integrity flag is specified.

# Open/attach the encrypted disks

cryptsetup luksOpen disk1.img disk1luks --key-file key.bin
cryptsetup luksOpen disk2.img disk2luks --key-file key.bin


# Create raid1:

mdadm \
  --create \
  --verbose --level 1 \
  --metadata=1.2 \
  --raid-devices=2 \
  /dev/md/mdtest \
  /dev/mapper/disk1luks \
  /dev/mapper/disk2luks


# Create a filesystem, add to LVM volume group, etc...

mkfs.ext4 /dev/md/mdtest

# Cool! Now you can 'scrub' the raid setup, which verifies
# the contents of each drive. Ordinarily detecting an error would
# be problematic, but since we are now using dm-integrity, the raid1
# *knows* which one has the correct data, and is able to fix it automatically. 
#
# To scrub the array:
#
#    $ echo check > /sys/block/md127/md/sync_action
#
# ... wait a while
#
#    $ dmesg | tail -n 30
#
# You should see 
#
#    [957578.661711] md: data-check of RAID array md127
#    [957586.932826] md: md127: data-check done.
#
#
# Let's simulate disk corruption:
# 
#    $ dd if=/dev/urandom of=disk2.img seek=30000 count=30 bs=1k conv=notrunc
#
# (this writes 30kB of random data into disk2.img)
#
#
# Run scrub again:
#
#    $ echo check > /sys/block/md127/md/sync_action
#
# ... wait a while
#
#    $ dmesg | tail -n 30
#
# Now you should see 
#	...
#    [959146.618086] md: data-check of RAID array md127
#    [959146.962543] device-mapper: crypt: INTEGRITY AEAD ERROR, sector 39784
#    [959146.963086] device-mapper: crypt: INTEGRITY AEAD ERROR, sector 39840
#    [959154.932650] md: md127: data-check done.
#
# But now if you run scrub yet again:
#    ...
#    [959212.329473] md: data-check of RAID array md127
#    [959220.566150] md: md127: data-check done.
#
# And since we didn't get any errors a second time, we can deduce that the invalid 
# data was repaired automatically.
#
# Great! We are done.
#
# --------
#
# If you don't need encryption, then you can use 'integritysetup' instead of cryptsetup. 
# It works in similar fashion: 
#
#    $ integritysetup format --integrity sha256 disk1.img
#    $ integritysetup format --integrity sha256 disk2.img
#    $ integritysetup open --integrity sha256 disk1.img disk1int
#    $ integritysetup open --integrity sha256 disk2.img disk2int
#    $ mdadm --create ...
#
# ...and so on. Though now you can detect and repair disk errors but have no protection
# against malicious cold-storage attacks. Data is also readable by anybody.
#
#	2018-03 NOTE: 
#
#       if you override the default --integrity value (whatever it is) during formatting, 
#       then you must specify it again when opening, like in the example above. For some 
#       reason the algorithm is not autodetected. I guess there is no header written onto 
#       disk like is with LUKS ?
#
# ----------
#
#  Read more:
#   https://fosdem.org/2018/schedule/event/cryptsetup/
#   https://gitlab.com/cryptsetup/cryptsetup/wikis/DMCrypt
#   https://gitlab.com/cryptsetup/cryptsetup/wikis/DMIntegrity
#   https://mirrors.edge.kernel.org/pub/linux/utils/cryptsetup/v2.0/v2.0.0-rc0-ReleaseNotes