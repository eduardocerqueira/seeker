#date: 2021-12-02T16:50:17Z
#url: https://api.github.com/gists/a2037723804ec0161e0e7bab4d6c38f2
#owner: https://api.github.com/users/sirredbeard

#!/bin/bash

echo 'Verify openSUSE Leap version 15.3'
cat /etc/os-release

echo 'Hardcode $releasever in Zypper repo files to 15.3'
sudo sed -i 's/15.3/${releasever}/g' /etc/zypp/repos.d/*.repo

echo 'Disable repos not yet available for openSUSE Leap 15.4 alpha. You should re-enable once 15.4 GA's.'
sudo sed -i 's/enabled=1/enabled=0/g' /etc/zypp/repos.d/repo-backports-update.repo
sudo sed -i 's/enabled=1/enabled=0/g' /etc/zypp/repos.d/repo-sle-update.repo
sudo sed -i 's/enabled=1/enabled=0/g' /etc/zypp/repos.d/repo-update.repo
sudo sed -i 's/enabled=1/enabled=0/g' /etc/zypp/repos.d/repo-update-non-oss.repo

echo 'Refresh respositories'
sudo zypper --releasever=15.4 refresh

echo 'Run distribution upgrade'
sudo zypper --releasever=15.4 dup --download-in-advance -y

echo 'Verify openSUSE Leap version 15.4'
cat /etc/os-release