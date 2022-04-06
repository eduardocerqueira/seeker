#date: 2022-04-06T17:12:30Z
#url: https://api.github.com/gists/15024f55f919079dd39cd718873cba19
#owner: https://api.github.com/users/lippirk

## diagnostics
# see remaining space
df -h /
# see what is using all the space
ncdu -x /

## purge
# see which programs you installed yourself
aptitude search '~i!~M'
# then
sudo apt remove --purge $prog

sudo apt autoremove --purge

# docker
docker system prune
docker system prune -a
docker volume prune

# flatpak
flatpak uninstall --unused

# snap
sudo snap set system refresh.retain=2
sudo rm -rf /var/cache/snapd
snap remove --purge $prog

# journalctl
sudo journalctl --rotate
sudo journalctl --vacuum-time=2d