#date: 2025-09-02T16:46:47Z
#url: https://api.github.com/gists/1a6850f1fd10ab57037a98817eb03170
#owner: https://api.github.com/users/danilogco

sudo nano /etc/NetworkManager/conf.d/10-ignore-docker.conf

# add it inside the file
# [keyfile]
# unmanaged-devices=interface-name:br-*;interface-name:veth*;interface-name:docker*;interface-name:lo

docker network prune