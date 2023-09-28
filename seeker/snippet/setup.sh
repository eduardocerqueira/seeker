#date: 2023-09-28T17:00:04Z
#url: https://api.github.com/gists/1844bbe0ee31a387a3262f2ee1ff4c64
#owner: https://api.github.com/users/Halu89


sudo apt install git -y
sudo apt install zsh -y
chsh -s $(which zsh)
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
echo "alias py=\"python3\"" >> .zshrc
echo "alias ll=\"ls -al\"" >> .zshrc