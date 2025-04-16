#date: 2025-04-16T16:53:55Z
#url: https://api.github.com/gists/659b6d85d7fa3806ce89fc4811375d3c
#owner: https://api.github.com/users/nero-dv

read -p "Enter drive path (ex: /dev/sdb): " drivepath
# wipe all partitions on 'drivepath'
sudo wipefs -a $drivepath
# create a gpt partition table, ext4 partition
sudo parted $drivepath mklabel gpt
sudo parted -a opt $drivepath mkpart primary ext4 0% 100%
sudo mkfs.ext4 drivepath\1