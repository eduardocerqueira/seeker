#date: 2022-07-19T16:55:56Z
#url: https://api.github.com/gists/113bc684937dca1b8028e104e07c305f
#owner: https://api.github.com/users/marcelbonnet

# install bash
/opt/homebrew/bin/brew install bash

# add this bash to the shells
echo "/opt/homebrew/bin/bash" > /etc/shells

# set my default shell
csh -s /opt/homebrew/bin/bash

# intialize PATH
touch ~/.bashrc
echo "PATH=\$PATH:/opt/homebrew/bin/" >> ~/.bashrc

# loads bashrc
cat << EOF > ~/.bash_profile
# Get the aliases and functions
if [ -f ~/.bashrc ]; then
    . ~/.bashrc
fi
EOF