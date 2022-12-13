#date: 2022-12-13T16:49:35Z
#url: https://api.github.com/gists/536383f11822bc0684e836503b1a587b
#owner: https://api.github.com/users/VinayakBector2002

#!/bin/bash
echo "Made with ❤ by Vinayak Bector"
echo "Installing dependencies for your React Native Application!"

#
# Check if Homebrew is installed
#
echo "Checking Brew ...."
which -s brew
if [[ $? != 0 ]] ; then
	printf "Brew is not installed! \nInstalling Brew \n"
	# Install Homebrew
	# https://github.com/mxcl/homebrew/wiki/installation
	/usr/bin/ruby -e "$(curl -fsSL https://raw.github.com/gist/323731)"
else
	printf "Brew is installed! \nUpdating Brew \n"
	#brew update
fi

# Helper Functions  - Response
# Install
# This will install the $depend

function Install {
    echo "Installing $1"
    brew install $1
}
# Helper Function - Response
# Update
# This will update the $depend

function Update {
    echo "Upgrading $1"
    brew upgrade $1
}

# Function -
# Response
# Checks if a depend is installed
#   if installed, then asks to update
#   else installs that depend

function AskRespo {
    read -p "Do you want to update $1? (y/n) " yn
    case $yn in 
	[yY] ) Update $1;;
	[nN] ) echo "Not Updating $1";;
	* ) echo "Invalid Response";
        AskRespo $1;;
    esac
}

# 
# Installing Rest of the dependencies
#
echo "Installing Dependancies ...."
for depend in git Node watchman ruby yarn android-platform-tools
do 
	which -s $depend
	if [[ $? != 0 ]] ; then
        Install $depend
	else
        # Depend already installed
    	echo "You already have $depend"
        AskRespo $depend
	fi
	echo ""
done
