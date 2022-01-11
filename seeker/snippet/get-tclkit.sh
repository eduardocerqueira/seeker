#date: 2022-01-11T17:19:37Z
#url: https://api.github.com/gists/3d3b9a39fca463ca9e16628e96877b5c
#owner: https://api.github.com/users/NyaMisty

#!/bin/sh
cp installbuilder/paks/linux-x64-noupx.pak tclkit
sed --in-place --null-data --expression='/^if {\[file isfile \[file join \$::tcl::kitpath main.tcl/{s/^./\x1A/}' tclkit
chmod +x tclkit