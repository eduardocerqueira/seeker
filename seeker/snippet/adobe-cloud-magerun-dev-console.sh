#date: 2021-12-22T17:03:06Z
#url: https://api.github.com/gists/650564d5ebe0c43c8c2a6b27248903de
#owner: https://api.github.com/users/vbuck

cd /tmp
curl -O https://files.magerun.net/n98-magerun2.phar
chmod +x ./n98-magerun2.phar
XDG_CONFIG_HOME="/tmp" ./n98-magerun2.phar dev:console --root-dir=$MAGENTO_CLOUD_DIR