#date: 2024-10-31T16:50:14Z
#url: https://api.github.com/gists/2caf44eafe6067be85e7a564f80da988
#owner: https://api.github.com/users/DonovanDiamond

#!/bin/bash

wget https://github.com/helix-editor/helix/releases/download/24.07/helix-24.07-x86_64-linux.tar.xz
tar -Jxvf helix-24.07-x86_64-linux.tar.xz
mkdir ~/bin
cd helix-24.07-x86_64-linux
cp hx ~/bin/
mkdir -p ~/.config/helix/
cp -r runtime ~/.config/helix/

cat << EOF > ~/.config/helix/config.toml
theme = "dark_plus"
[editor]
true-color = true

[editor.cursor-shape]
insert = "bar"
normal = "block"
select = "underline"

[editor.file-picker]
hidden = false
EOF
