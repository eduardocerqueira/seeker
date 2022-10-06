#date: 2022-10-06T17:24:22Z
#url: https://api.github.com/gists/56c2df026d549ee10c32b2daaca80ac4
#owner: https://api.github.com/users/crazyoptimist

#!/bin/bash
curl -s https://api.github.com/repos/sharkdp/bat/releases/latest \
| grep -v ".sha256" \
| grep browser_download_url
curl -SL https://github.com/sharkdp/bat/releases/download/v0.22.1/bat_0.22.1_amd64.deb -o bat.deb
sudo dpkg -i bat.deb
rm bat.deb