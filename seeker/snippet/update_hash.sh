#date: 2022-04-13T16:50:36Z
#url: https://api.github.com/gists/7930aea52513c5c0052b0448853a4bf7
#owner: https://api.github.com/users/tsundokul

# Remove legacy keyshit 
sudo rm /etc/apt/trusted.gpg
# Run apt update to get the deprecated gpg key hashes printed
sudo apt update
# Err:13 https://deb.torproject.org/torproject.org stretch InRelease
#  The following signatures couldn't be verified because the public key is not available: NO_PUBKEY 74A941BA219EC810

# replace [HASH] eg. F57D4F59BD3DF454
# [NAME] must be unique eg. tor.gpg, sublime.gpg etc
cd /tmp
curl 'https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x[HASH]' | gpg --dearmor > [NAME].gpg
sudo install -o root -g root -m 644 [NAME].gpg /etc/apt/trusted.gpg.d/
