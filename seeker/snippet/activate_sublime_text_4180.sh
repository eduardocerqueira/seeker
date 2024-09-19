#date: 2024-09-19T16:39:39Z
#url: https://api.github.com/gists/9a19389de83b1d47863520c68548d441
#owner: https://api.github.com/users/grilled-snakehead

#!/usr/bin/env bash

sudo cp /opt/sublime_text/sublime_text ./sublime_text.old
sudo sed -i 's/\x80\x79\x05\x00\x0F\x94\xC2/\xC6\x41\x05\x01\xB2\x00\x90/' /opt/sublime_text/sublime_text