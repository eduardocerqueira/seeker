#date: 2024-01-05T17:05:09Z
#url: https://api.github.com/gists/1138627f706544e8de7ea95bd5d2b0a5
#owner: https://api.github.com/users/dmtzs

#!/usr/bin/bash

ssh-keygen -t ed25519 -C "your_email"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
xclip -selection clipboard < ~/.ssh/id_ed25519.pub
echo "public key added to clipboard"