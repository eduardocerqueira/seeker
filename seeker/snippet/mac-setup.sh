#date: 2022-02-01T17:07:35Z
#url: https://api.github.com/gists/d23be01863e112db81486d64a1ece90d
#owner: https://api.github.com/users/hassanejaz

#!/bin/bash

# Non-scripted Tasks:
#   - Configure device name in Preferences > Sharing
#   - Enable Remote Login & Remote Management in Preferences > Sharing
#   - Enable automatic login/disable password after sleep in Preferences > Security & Privacy > General
#   - Disable screensaver/sleep in Preferences > Energy Saver
#   - Disable spotlight indexing of home directory
#   - Add a runner in GitHub UI to grab your token https://github.com/<org>/<repo>/settings/actions/add-new-runner

# If M1 enable Rosetta
softwareupdate --install-rosetta

# Disable spotlight indexing service
sudo mdutil -a -i off

# Install homebrew, this should also prompt to install XCode
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"

# Install extra taps
brew tap homebrew/cask 
brew tap homebrew/cask-versions 
brew tap homebrew/services

# Install utilities
brew update
brew install git git-lfs hub
brew install --cask docker google-chrome firefox microsoft-edge 
git lfs install

# Install nvm (see https://github.com/nvm-sh/nvm#installing-and-updating)
touch ~/.zshrc
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.37.2/install.sh | bash

# Setup npm packages
nvm install v14
npm install -g yarn

# Setup GitHub runner
mkdir actions-runner && cd actions-runner
curl -O -L https://github.com/actions/runner/releases/download/v2.263.0/actions-runner-osx-x64-2.263.0.tar.gz
tar xzf ./actions-runner-osx-x64-2.263.0.tar.gz
./config.sh --url https://github.com/XXXXX/XXXXX --token XXXX
./svc.sh install
./svc.sh start