#date: 2022-03-01T17:03:49Z
#url: https://api.github.com/gists/d885da5b7fb655799bbcdc59f297350c
#owner: https://api.github.com/users/vbalagovic

#!/bin/bash

echo "rootqed=\"$PWD\"" | cat - qedaliases > localaliases

echo "source $PWD/localaliases" >> ~/.zshrc
echo "source $PWD/localaliases" >> ~/.bashrc

echo -n -e | ./init.sh networks