#date: 2024-06-05T17:07:47Z
#url: https://api.github.com/gists/ce509c2de8a180aef4b3d833f2812d6d
#owner: https://api.github.com/users/Anonymate054

Configuración terminal

sudo apt install tilix

sudo apt install zsh

chsh -s $(which zsh)

exec zsh

sh -c "$(wget https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"

git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k

nano ~/.zshrc

#Cambiar: ZSH_THEME="powerlevel10k/powerlevel10k"

curl -L -O https://github.com/romkatv/powerlevel10k-media/raw/master/MesloLGS%20NF%20Regular.ttf
curl -L -O https://github.com/romkatv/powerlevel10k-media/raw/master/MesloLGS%20NF%20Bold.ttf
curl -L -O https://github.com/romkatv/powerlevel10k-media/raw/master/MesloLGS%20NF%20Italic.ttf
curl -L -O https://github.com/romkatv/powerlevel10k-media/raw/master/MesloLGS%20NF%20Bold%20Italic.ttf

# Reconfigurar: p10k configure