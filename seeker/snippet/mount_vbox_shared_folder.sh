#date: 2024-12-05T17:12:02Z
#url: https://api.github.com/gists/5cc903d166dcdc0cc7c545608454ec28
#owner: https://api.github.com/users/jcrtexidor

# ~/.cmd/mount_vbox_shared_folder.sh

mkdir -p /media/shared_folder
if [ $(mount | grep -q "shared_folder") ]; then
        mount -t vboxsf shared_folder /media/shared_folder
fi
