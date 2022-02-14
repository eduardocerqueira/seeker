#date: 2022-02-14T17:01:53Z
#url: https://api.github.com/gists/29f0db6fbc6e3b8381f128ceebfa7854
#owner: https://api.github.com/users/megapod

# .bash_profile

# Get the aliases and functions
if [ -f ~/.bashrc ]; then
	. ~/.bashrc
fi

# User specific environment and startup programs

PATH=$PATH:$HOME/.local/bin:$HOME/bin

export PATH

if [ -f ~/login_script.sh ]; then
	sh ~/login_script.sh
fi
