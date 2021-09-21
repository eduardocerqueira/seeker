#date: 2021-09-21T17:04:33Z
#url: https://api.github.com/gists/e50bbeebb337901e99aea3b2c5ccc350
#owner: https://api.github.com/users/brewpirate

#!/bin/sh

# Install Brew!
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# == Git Shit =====================
# https://formulae.brew.sh/formula/git
brew install git

# https://formulae.brew.sh/formula/git-utils#default
brew install git-utils

# https://formulae.brew.sh/formula/git-flow-avh#default
brew install git-flow-avh


# == Core Dev ====================
# https://formulae.brew.sh/formula/coreutils
brew install coreutils

# AWS CLI
# https://formulae.brew.sh/formula/awscli
brew install awscli

# https://formulae.brew.sh/formula/node
brew install node

# https://formulae.brew.sh/formula/curl#default
brew install curl

# https://formulae.brew.sh/cask/google-chrome#default
brew install --cask google-chrome	

# https://formulae.brew.sh/cask/firefox#default
brew install --cask firefox

# https://formulae.brew.sh/cask/mysqlworkbench#default
brew install --cask mysqlworkbench

# https://formulae.brew.sh/cask/sequel-pro#default
brew install --cask sequel-pro		

# https://formulae.brew.sh/cask/cyberduck#default
brew install --cask cyberduck

# https://formulae.brew.sh/cask/gitkraken#default
# brew install --cask gitkraken		

# https://formulae.brew.sh/cask/lastpass#default
brew install --cask lastpass		

# https://formulae.brew.sh/cask/postman#default
brew install --cask postman

# https://formulae.brew.sh/formula/openvpn
brew install openvpn

#https://formulae.brew.sh/cask/iterm2#default
brew install --cask iterm2

# == Lang Tools ====================
# PHP - ...
# https://formulae.brew.sh/formula/composer#default
# brew install composer


# == Docker ====================
# https://formulae.brew.sh/formula/docker#default
brew install --cask docker

# https://formulae.brew.sh/formula/docker-compose#default
brew install docker-compose

# http://docker-sync.io/
sudo gem install docker-sync


# https://docker-sync.readthedocs.io/en/latest/getting-started/installation.html#advanced-optional
brew install unison
brew install rsync

# Default 
# brew install eugenmayer/dockersync/unox
# Better Sync?
brew install autozimu/homebrew-formulas/unison-fsmonitor


# == IDE ====================
# Uncomment to install the IDE(s) of choice.

# https://formulae.brew.sh/cask/phpstorm#default
#brew install --cask phpstorm

# https://formulae.brew.sh/cask/visual-studio-code#default
#brew install --cask visual-studio-code

# https://formulae.brew.sh/cask/sublime-text#default
#brew install --cask sublime-text

# https://formulae.brew.sh/cask/webstorm#default
#brew install --cask webstorm


# == Business Tools ==============
# https://formulae.brew.sh/cask/ringcentral#default
brew install --cask ringcentral

# https://formulae.brew.sh/cask/ringcentral-meetings#default
brew install --cask ringcentral-meetings

# https://formulae.brew.sh/cask/microsoft-teams#default
brew install --cask microsoft-teams		

# https://formulae.brew.sh/cask/microsoft-outlook#default
brew install --cask microsoft-outlook

# == Optional ====================

# SqlSrv Tools for Mac
# brew install --cask azure-data-studio

# https://formulae.brew.sh/formula/htop
brew install htop

# https://formulae.brew.sh/cask/cakebrew#default
brew install --cask cakebrew

# https://formulae.brew.sh/cask/meld#default
#brew install --cask meld
#brew install --cask dash	
#brew install --cask spotify