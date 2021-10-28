#date: 2021-10-28T17:09:28Z
#url: https://api.github.com/gists/2d3ce7d8b15ec0557c1f5932b276b654
#owner: https://api.github.com/users/Qetlin

#!/usr/bin/env bash

##################################
# Install command line dev tools #
##################################
/usr/bin/xcode-select -p > /dev/null 2>&1
if [ $# != 0 ]; then
  xcode-select --install
  sudo xcodebuild -license accept
fi

###############################
# Do not allow rootless login #
###############################
ROOTLESS_STATUS=$(/usr/bin/csrutil status | awk '/status/ {print $5}' | sed 's/\.$//')
if [[ $ROOTLESS_STATUS == "enabled" ]]; then
  echo "csrutil (\"rootless\") is enabled. please disable in boot screen and run again!"
  exit 1
fi

#####################
# Turn on FileVault #
#####################
FILEVAULT_STATUS=$(fdesetup status)
if [[ $FILEVAULT_STATUS != "FileVault is On." ]]; then
  echo "FileVault is not turned on. Please encrypt your hard disk!"
fi

#################################
# Change default shell to `zsh` #
#################################
chsh -s /bin/zsh

#################################
# Setup ssh scripts/directories #
#################################
mkdir -p ~/.ssh
sudo chmod 755 ~ && sudo chmod 700 ~/.ssh && sudo chmod 600 ~/.ssh/id* && sudo chmod 644 ~/.ssh/id*.pub

######################
# vim configurations #
######################
curl -L https://gist.githubusercontent.com/vraravam/2d8654cb21bfe506a64a05a49268d9de/raw/8d2706c756f415fa8d1f7de99abf1e65804c53d5/.vimrc -o ~/.vimrc

#####################
# Install oh-my-zsh #
#####################
curl -L http://install.ohmyz.sh | sh

curl -L https://gist.githubusercontent.com/vraravam/657c3b94d1b04bacd2b6a38c22d6ec56/raw/5c7cd232221fd0ee12ac3cf7d5ef193ad3405128/.zshrc -o ~/.zshrc
curl -L https://gist.githubusercontent.com/vraravam/5d824ccf6353f3449f5a80c6d06ff2c0/raw/032c5ac2c0ba32c0b1e157316918f8ddb50e5b76/.zshrc.pre-oh-my-zsh -o ~/.zshrc.pre-oh-my-zsh

#######################################################################
# Install custom plugins for auto-suggestions and syntax highlighting #
#######################################################################
git clone git://github.com/zsh-users/zsh-autosuggestions "${ZSH_CUSTOM:-~/.oh-my-zsh/custom}"/plugins/zsh-autosuggestions
git clone git://github.com/zsh-users/zsh-syntax-highlighting.git "${ZSH_CUSTOM:-~/.oh-my-zsh/custom}"/plugins/zsh-syntax-highlighting

# <RESTART TERMINAL APP>
################################
# Prep for installing homebrew #
################################
sudo mkdir -p /usr/local/tmp /usr/local/repository /usr/local/plugins
sudo chown -fR "$(whoami)":admin /usr/local
sudo rm -rf ~/.gnupg  # to delete gpg keys that might have been generated from an older version of gpg

#######################################
# Install homebrew (on empty machine) #
#######################################
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
# This list is the 'first-set' of installations from brew, followed by the others inside the Brewfile. Done as an extra bootstrap step - so as to avoid installation issues when running the full bundle
brew install direnv zsh coreutils autoconf automake readline openssl libyaml libxslt libtool unixodbc gnupg git

curl -L https://gist.githubusercontent.com/vraravam/8c9eae91a3750bed86b81e3a4711f842/raw/c11de0c195f65110c2ed3eac6dfdd623fb09160a/Brewfile -o Brewfile
brew bundle   # You can run this here - but be prepared to have errors reported due to missing java

# <RESTART TERMINAL APP>

# <At this point, you will need to login into the gmail account, retrieve the keybase paper key and install keybase app>

#######################
# Clone the home repo #
#######################
mkdir ~/tmp
git clone keybase://private/avijayr/vijay ~/tmp
mv ~/tmp/.git ~
rm -rf ~/tmp

##########################
# Clone the Library repo #
##########################
mkdir ~/tmp
git clone keybase://private/avijayr/Library ~/tmp
mv ~/tmp/.git ~/Library
rm -rf ~/tmp

# <At this point, you will need to manually reconcile which files to be kept/deleted from the Library repo>

#########################################
# Fix /etc/hosts file to block facebook #
#########################################
sudo cp ~/.bin/etc.hosts /etc/hosts

############################################################################################################################
# Checkout some of the common dotfiles (these should not have any modifications/conflicts with what is in the remote repo) #
############################################################################################################################
git checkout .aliases .anyconnect .bundle .config .dbshell .dlv .eclipse .editorconfig .erdconfig .gitconfig .gitmodules .gnupg .gsutil .helm .iex.exs .ionic .kube .mix .notify-osd .openvpn-connect.json .p2 .profile .sqldeveloper .ssh .tldrc .tmuxp .tool-versions .tooling .vscode-insiders .vscode-react-native .zlogin ".*lint*" ".*rc" ".ansible*" ".bash*" ".docker*" ".erl*" ".fzf*" ".gitignore*" ".hgignore*" ".psql*" ".tmux*" ".v8flags*" ".zshrc*" "*history" Brewfile OSX Personal

##################################################
# Manually diff, triage and checkout other files #
##################################################
# <Checkout those files that you want in addition to the above; resolve differences in existing files>

################################################
# Reapply permissions for ssh  and gnupg files #
################################################
sudo chmod 600 ~/.ssh/id* && sudo chmod 644 ~/.ssh/id*.pub
sudo chmod 600 ~/.gnupg/* && chmod 700 ~/.gnupg && rm -rf .gnupg/crls.d

# Note: VS Code and Tmux settings are in a separate git repository
git clone https://gitlab.com/avijayr/vscode.git ~/.vscode
git clone https://github.com/gpakosz/.tmux.git ~/.tmux

###########################################################################
# Install ASDF package manager - requires changes present in the `.zshrc` #
###########################################################################
# Note: installing via homebrew is broken. To check see `ruby -v`
export ASDF_DIR="$HOME/.asdf"
rm -rf ~/.asdf
git clone https://github.com/asdf-vm/asdf.git ~/.asdf --branch v0.6.3

# <RESTART TERMINAL APP>

# Note: immediately after you install asdf, you will also have to echo certain lines into bash_profile or zshrc - please see here: https://github.com/asdf-vm/asdf (since i have that incorporated already, i dont have it in this script)
asdf plugin-add erlang https://github.com/asdf-vm/asdf-erlang.git
asdf plugin-add elixir https://github.com/asdf-vm/asdf-elixir.git
asdf plugin-add ruby https://github.com/asdf-vm/asdf-ruby.git
asdf plugin-add java https://github.com/skotchpine/asdf-java
asdf plugin-add rust https://github.com/code-lever/asdf-rust.git
asdf plugin-add golang https://github.com/kennyp/asdf-golang.git
asdf plugin-add python https://github.com/danhper/asdf-python.git

# Imports Node.js release team's OpenPGP keys to main keyring
asdf plugin-add nodejs https://github.com/asdf-vm/asdf-nodejs.git
# TODO: Not sure why $ASDF_DIR is not working
bash ~/.asdf/plugins/nodejs/bin/import-release-team-keyring

# Note: Since I have already configured the `.tool-versions` file, we don't need to install each language separately
asdf install

# asdf plugin-list
# asdf plugin-update --all

# To list all versions of a plugin
# asdf list-all erlang

# To reshim the plugin
# asdf reshim erlang

# To set the global version
# asdf global erlang 20.3.8

# <RESTART TERMINAL APP>

# To install the latest versions of the hex, rebar and phoenix packages
mix local.hex --force && mix local.rebar --force
mix archive.install hex phx_new 1.4.1

#################################
# Rerun after java is installed #
#################################
brew bundle  # This will use the Brewfile to install/upgrade all the libraries and apps and their dependencies

# Post-install step for postgres
createuser -s postgres
psql -U postgres -c "CREATE ROLE \"postgres\" WITH PASSWORD 'postgres' LOGIN CREATEDB;"
# Post-install step for fzf
"$(brew --prefix)"/opt/fzf/install   # TODO: See how this can be accomplished within the Brewfile syntax
# since we are using 'code-insiders' only, symlink it to 'code' for ease of typing
ln -sf "$(brew --prefix)"/bin/code-insiders "$(brew --prefix)"/bin/code

#########
# tmuxp #
#########
pip3 install --user tmuxp

# To get a specific version of imagemagick - TODO: check how this can be incorporated into the Brewfile
#brew uninstall -f imagemagick
#brew install https://raw.githubusercontent.com/Homebrew/homebrew-core/6f014f2b7f1f9e618fd5c0ae9c93befea671f8be/Formula/imagemagick.rb
#brew pin imagemagick

# default gems
# libffi needs some missing gcc lib linked. for more info, see here: https://sukhbinder.wordpress.com/2016/01/28/solving-ld-library-not-found-issues-after-upgrading-to-os-x-el-capitain/
cd /usr/local/lib && sudo ln -s ../../lib/libSystem.B.dylib libgcc_s.10.4.dylib
gem install bundler rake rubocop bundler-audit ruby_audit brakeman metric_fu mry

# <RESTART TERMINAL APP>

# <FOLLOW instructions on https://github.com/vic/asdf-link to install for jdk>
#
# For eg:
# ls /Library/Java/JavaVirtualMachines/
# asdf install jdk 1.8.0
# ls /Library/Java/JavaVirtualMachines/jdk1.8.0_152.jdk/Contents/Home/bin/*
# ln -vs /Library/Java/JavaVirtualMachines/jdk1.8.0_152.jdk/Contents/Home/bin/* /Users/vijay/.asdf/installs/jdk/1.8.0/bin
# asdf reshim jdk

# Enabling history for iex shell (might need to be done for each erl that is installed via asdf)
rm -rf tmp
mkdir -p tmp
cd tmp
git clone https://github.com/ferd/erlang-history.git
cd erlang-history
sudo make install
cd ../..
rm -rf tmp

if [[ "$OSTYPE" = darwin* ]] ; then
  # rm -rf ~/Library/Application\ Support/Skype && ln -sf ~/Dropbox/Skype ~/Library/Application\ Support/Skype

  rm -rf ~/Library/Application\ Support/Mozilla ~/Library/Mozilla

  # mkdir -p ~/Library/Application\ Support/Google/Chrome
  # rm -rf ~/Library/Application\ Support/Google/Chrome && ln -sf ~/Personal/vijay/ChromeProfile ~/Library/Application\ Support/Google/Chrome

  mkdir -p ~/Library/Application\ Support/Firefox ~/Personal/vijay/FirefoxProfile
  rm -rf ~/Library/Application\ Support/Firefox/Crash\ Reports ~/Library/Application\ Support/Firefox/Profiles
  rm -rf ~/Library/Application\ Support/Firefox/profiles.ini && ln -sf ~/Personal/vijay/profiles.ini.ff ~/Library/Application\ Support/Firefox/profiles.ini
  touch ~/Library/Application\ Support/Firefox/ignore-dev-edition-profile

  mkdir -p ~/Library/Application\ Support/TorBrowser-Data/Browser
  rm -rf ~/Library/Application\ Support/TorBrowser-Data/Browser/profiles.ini && ln -sf ~/Personal/vijay/profiles.ini.ff ~/Library/Application\ Support/TorBrowser-Data/Browser/profiles.ini

  mkdir -p ~/Library/Thunderbird ~/Personal/vijay/ThunderbirdProfile
  rm -rf ~/Library/Thunderbird/Crash\ Reports ~/Library/Thunderbird/Profiles
  rm -rf ~/Library/Thunderbird/profiles.ini && ln -sf ~/Personal/vijay/profiles.ini.tb ~/Library/Thunderbird/profiles.ini
fi

# Sencha download (not present in homebrew) - If I continue with contribution on rambox
# curl -o ~/Downloads/SenchaCmd-6.5.3.6-osx.app.zip http://cdn.sencha.com/cmd/6.5.3.6/jre/SenchaCmd-6.5.3.6-osx.app.zip
# open ~/Downloads/SenchaCmd-6.5.3.6-osx.app.zip

# ln -s "$(brew --prefix)"/include/ImageMagick/wand "$(brew --prefix)"/include/wand && ln -s "$(brew --prefix)"/include/ImageMagick/magick "$(brew --prefix)"/include/magick

# npm install -g npm@latest eslint eslint-config-defaults eslint-config-google eslint-plugin-backbone eslint-plugin-html

# vagrant plugin install vagrant-vbguest

# TO install the ansible-elixir-stack tools for releasing elixir apps
# ansible-galaxy install HashNuke.elixir-stack
