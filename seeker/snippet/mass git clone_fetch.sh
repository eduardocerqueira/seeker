#date: 2024-09-27T17:08:40Z
#url: https://api.github.com/gists/4adbcc7ad5875c4a388160daf1dae2ff
#owner: https://api.github.com/users/damdam-s

for line in $(ls -1d */);
do
pushd $line && git fetch --all && popd;
done


------
#!/bin/bash

test_file=$(./test_github.py)

while IFS= read -r line
do
    echo $line
    [ -d $line ] && pushd $line && git fetch --all && popd || git clone "git@github.com:camptocamp/$line.git" && pushd $line && git fetch --all && popd
done < REPOS

