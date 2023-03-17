#date: 2023-03-17T17:06:19Z
#url: https://api.github.com/gists/365197bce3cf51149795982e5652fb8c
#owner: https://api.github.com/users/yiitz

sudo rm -rfv /Library/Caches/com.apple.iconservices.store; sudo find /private/var/folders/ \( -name com.apple.dock.iconcache -or -name com.apple.iconservices \) -exec rm -rfv {} \; ; sleep 3;sudo touch /Applications/* ; killall Dock; killall Finder