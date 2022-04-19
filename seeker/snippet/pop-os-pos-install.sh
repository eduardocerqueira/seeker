#date: 2022-04-19T17:11:33Z
#url: https://api.github.com/gists/37b0e8a2f97b92d38833d52947c87def
#owner: https://api.github.com/users/jesherdevsk8

#!/bin/bash

### Variáveis
PROGRAMAS_APT=(
    git
    git-flow
    google-chrome-stable
    deepin-terminal
    deepin-screen-recorder
    gnome-tweak-tool
    snapd
    ubuntu-restricted-extras
    unrar
    gimp
    curl
    inkscape
)

PROGRAMAS_SNAP=(
    postman
    spotify
)

PROGRAMAS_SNAP_CLASSIC=(
    slack
    gitkraken
)

PROGRAMAS_PARA_REMOVER=(
    firefox
    firefox-locale-ar
    firefox-locale-de
    firefox-locale-en
    firefox-locale-es
    firefox-locale-fr
    firefox-locale-it
    firefox-locale-ja
    firefox-locale-pt
    firefox-locale-ru
    firefox-locale-zh-hans
    firefox-locale-zh-hant
    flatpak
    geary
    libreoffice
    libreoffice-base-core
    libreoffice-common
    libreoffice-core
    libreoffice-help-common
    libreoffice-style-colibre
    libreoffice-style-tango
    gnome-screenshot
)

disable_locks() {
    sudo rm /var/lib/apt/lists/lock
    sudo rm /var/lib/dpkg/lock
    sudo rm /var/lib/dpkg/lock-frontend
    sudo rm /var/cache/apt/archives/lock
}

enable_locks() {
    sudo dpkg --configure -a
}

update_upgrade() {
    sudo apt update && sudo apt upgrade -y && sudo apt dist-upgrade -y
}

remove_clean() {
    sudo apt autoremove -y && sudo apt autoclean -y
}

### Pré-Execução
update_upgrade


## Remover travas eventuais do apt ##
disable_locks

## Atualizar o repositório
update_upgrade

## Instalar o ttf-mscorefonts-installer para impedir a confirmação mais à frente
sudo apt install ttf-mscorefonts-installer -y

## Requisitos do Chrome
wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add - 
sudo sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'

### Execução

#INSTALAR JETBRAINS TOOLBOX

## Atualizar o repositório
update_upgrade

## Remover programas no apt
for nome_do_programa in ${PROGRAMAS_PARA_REMOVER[@]}; do
  sudo apt remove "$nome_do_programa" -y
done

update_upgrade
remove_clean

## Instalar programas no apt
for nome_do_programa in ${PROGRAMAS_APT[@]}; do
    sudo apt install "$nome_do_programa" -y
done

## Instalar programas pelo snap
for nome_do_programa in ${PROGRAMAS_SNAP[@]}; do
    sudo snap install "$nome_do_programa"
done

## Instalar programas pelo snap classic
for nome_do_programa in ${PROGRAMAS_SNAP[@]}; do
    sudo snap install --classic "$nome_do_programa"
done

## Instalar fontes Truetype e Opentype
##INSTALAR FIRA CODE

### Pós-execução

## Finalizar, atualizar e limpar
update_upgrade
remove_clean
enable_locks