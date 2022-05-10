#date: 2022-05-10T17:14:24Z
#url: https://api.github.com/gists/457342277a98c594bc00fd6e52ee2383
#owner: https://api.github.com/users/jrc03c

# assuming a partition already exists, apply LUKS formatting to it
sudo cryptsetup luksFormat /dev/whatever

# open it
sudo cryptsetup luksOpen /dev/whatever mapping_name

# format it to ext4 (or whatever)
sudo mkfs.ext4 /dev/mapper/mapping_name

# mount it
sudo mount /dev/mapper/mapping_name /some/path

# unmount it
sudo unmount /some/path

# close it
sudo cryptsetup luksClose mapping_name