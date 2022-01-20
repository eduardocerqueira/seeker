#date: 2022-01-20T16:55:45Z
#url: https://api.github.com/gists/27dbf3db740c778b01c7642a89fbfc40
#owner: https://api.github.com/users/kis9a

function mac_cleanup_storage() {
  sudo rm -rf ~/Library/Developer/Xcode/DerivedData/*
  sudo rm -rf ~/Library/Developer/Xcode/Archives/*
  sudo rm -rf ~/Library/Caches/*
  sudo rm -rf ~/Library/Logs/iOS\ Simulator
  sudo rm -rf ~/Library/Developer/Xcode/iOS\ DeviceSupport/*
  sudo launchctl unload /System/Library/LaunchDaemons/com.apple.dynamic_pager.plist
  sudo launchctl load /System/Library/LaunchDaemons/com.apple.dynamic_pager.plist
  sudo rm -rf ~/Library/Application\ Support/Adobe/Common/Media\ Cache/* ~/Library/Application\ Support/Adobe/Common/Media\ Cache\ Files/*
  sudo rm -rf /System/Library/Caches/* /Library/Caches/* ~/Library/Caches/*
  sudo rm -r /.DocumentRevisions-V100/.cs
  sudo mdutil -i off /
  sudo rm -rf /.Spotlight-V100/
  sudo mdutil -E -i on /
  sudo rm -r /private/var/vm/sleepimage
}