#date: 2022-01-12T17:15:11Z
#url: https://api.github.com/gists/350d641f92671313074121cfb7a44eaf
#owner: https://api.github.com/users/Alfredosavi

#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

# ----------------------------- VARIÁVEIS ----------------------------- #
# PPA_LIBRATBAG="ppa:libratbag-piper/piper-libratbag-git"
# URL_GOOGLE_CHROME="https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb"

DIRETORIO_DOWNLOADS="$HOME/Downloads/programas"
PORTAINER_IMG_LABEL=":linux-arm"

PACOTES_PARA_INSTALAR=(
  raspberrypi-kernel # Docker
  raspberrypi-kernel-headers # Docker
  libffi-dev # Docker-Compose
  libssl-dev # Docker-Compose
  python3-dev # Docker-Compose
  python3 # Docker-Compose
  python3-pip # Docker-Compose
)

PROGRAMAS_PARA_INSTALAR=(
  zsh
  ffmpeg
  code # Visual Studio Code
  gparted
)


# ----------------------------- REQUISITOS ----------------------------- #
## Removendo travas eventuais do apt ##
sudo rm -f /var/lib/dpkg/lock-frontend
sudo rm -f /var/cache/apt/archives/lock


# ----------------------------- EXECUÇÃO ----------------------------- #
## Atualizando o repositório depois da adição de novos repositórios ##
sudo apt update && sudo apt upgrade -y

echo "---------------------------"


## Download e instalaçao de programas externos ##
mkdir -p "$DIRETORIO_DOWNLOADS"
# wget -c "$URL_GOOGLE_CHROME"       -P "$DIRETORIO_DOWNLOADS"


## Instalando pacotes .deb baixados na sessão anterior ##
if find DIRETORIO_DOWNLOADS -mindepth 1 | read; then
  sudo dpkg -i $DIRETORIO_DOWNLOADS/*.deb
else
  echo "DIR EMPTY!"
fi


# Instalar pacotes de dependencias via APT
for nome_do_pacote in ${PACOTES_PARA_INSTALAR[@]}; do
  if ! dpkg -l | grep -q $nome_do_pacote; then # Só instala se já não estiver instalado
    sudo apt install "$nome_do_pacote" -y
  else
    echo "[INSTALADO] - $nome_do_pacote"
  fi
done


# Instalar programas via APT
for nome_do_programa in ${PROGRAMAS_PARA_INSTALAR[@]}; do
  if ! dpkg -l | grep -q $nome_do_programa; then # Só instala se já não estiver instalado
    sudo apt install "$nome_do_programa" -y
  else
    echo "[INSTALADO] - $nome_do_programa"
  fi
done


# ---------------------INSTALANDO DOCKER------------------------ #
curl -sSL https://get.docker.com | sh
sudo usermod -aG docker pi
echo "Docker instalado com sucesso!"

## Instalando o Docker-Compose ##
sudo pip3 install docker-compose
echo "Docker-Compose instalado com sucesso!"

sudo systemctl enable docker

## Instalando Portainer
sudo docker pull portainer/portainer-ce/${PORTAINER_IMG_LABEL}

sudo docker run -d -p 9000:9000 --name=portainer --restart=always -v /var/run/docker.sock:/var/run/docker.sock -v portainer_data:/data portainer/portainer-ce:latest
echo "[INFO] Portainer listening on port 9000"

# -----------------CONFIGURANDO ZSH----------------------- #
chsh -s $(which zsh) # Definindo o zsh como padrão

sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

## Plugins ZSH
cd ~/.oh-my-zsh/plugins

git clone https://github.com/zsh-users/zsh-syntax-highlighting.git # zsh-synstax-highlighting 
git clone https://github.com/zsh-users/zsh-autosuggestions # zsh-autosuggestions

echo "source ${(q-)PWD}/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh" >> ${ZDOTDIR:-$HOME}/.zshrc
echo "source ${(q-)PWD}/zsh-autosuggestions/zsh-autosuggestions.zsh" >> ${ZDOTDIR:-$HOME}/.zshrc

source ./zsh-syntax-highlighting/zsh-syntax-highlighting.zsh
source ./zsh-autosuggestions/zsh-autosuggestions.zsh




# ----------------------RETIRANDO NO PASSWORD ROOT------------------------ #
sudo rm -f /etc/sudoers.d/010-pi-nopasswd
sudo adduser pi sudo


# ----------------------------- PÓS-INSTALAÇÃO ----------------------------- #
## Finalização, atualização e limpeza##
echo "Atualizando e limpeza..."
sudo apt update && sudo apt dist-upgrade -y
sudo apt autoclean
sudo apt autoremove -y

rm -rf $DIRETORIO_DOWNLOADS

echo "Pronto! :)"