#date: 2021-10-20T17:15:55Z
#url: https://api.github.com/gists/b69ee2f630a32ec0f1f2093e145312e0
#owner: https://api.github.com/users/papilip

#--------------------------------
# Vérifier le nom de l’ordinateur
#--------------------------------
sudo scutil --set HotsName "mon-ordi"

#--------------------------------
# Installation de HomeBrew et d’iterm2 pour quitter au plus vite Terminal
#--------------------------------
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew install iterm2

#--------------------------------
# Installation de Oh My Zsh
#--------------------------------
sh -c "$(curl -fsSL https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"

#--------------------------------
# Installer les application en ligne de commande pour les geek
#--------------------------------
brew install asdf archey autojump git ghostscript gzip imagemagick htop httrack mariadb mas midnight-commander nginx node p7zip php phpmyadmin postgresql python rsync sha3sum speedtest-cli sqlite trash tree wget wifi-password yarn youtube-dl

#--------------------------------
# Pour les autres
#--------------------------------
brew install asdf autojump git p7zip trash tree wget wifi-password

#--------------------------------
# Installer les application graphiques
#--------------------------------
brew install adobe-acrobat-reader adoptopenjdk8 bettertouchtool chromium discord dropbox firefox gimp google-chrome inkscape jdownloader jitsi-meet opera rectangle scribus signal simple-comic skype speedcrunch teamviewer telegram-desktop twitch vlc zoom

#--------------------------------
# Les application graphiques pour developpeur
#--------------------------------
brew install dbeaver-community electron gcollazo-mongodb gitahead gitfiend github pdf-images pgadmin4 sublime-merge sublime-text vscodium

#--------------------------------
# Pour installer AsciiDocFX
#--------------------------------
sudo spctl --master-disable
brew install asciidocfx
sudo spctl --master-enable

#--------------------------------
# Les services si besoin
#--------------------------------
brew services start mariadb
brew services start nginx
brew services start php
brew services start postgresql

#--------------------------------
# Mise en oeuvre d’ASDF
#--------------------------------
asdf plugin-add ruby https://github.com/asdf-vm/asdf-ruby.git
asdf install ruby latest && asdf global ruby 3.0.2
asdf plugin-add crystal https://github.com/asdf-community/asdf-crystal.git
asdf install crystal latest && asdf global crystal 1.1.1
