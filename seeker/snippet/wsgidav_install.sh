#date: 2025-03-28T16:43:41Z
#url: https://api.github.com/gists/a604585b40e6927cf340bdf3f781e55c
#owner: https://api.github.com/users/shavedbroom

# Warning: don't use this for sensitive data, there's no SSL or authentication
# for ubuntu 20.04
# only allow access from another server with a dedicated ip
ufw allow from x.x.x.x
ufw enable
apt-get update
apt-get upgrade
apt-get install nginx-full
mkdir -p /var/dav/webdav_root
sudo chown -R www-data:www-data /var/dav/
rm /etc/nginx/sites-available/default
rm /etc/nginx/sites-enabled/default
# copy over webdav.conf
nano /etc/nginx/conf.d/webdav.conf