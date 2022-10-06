#date: 2022-10-06T17:14:41Z
#url: https://api.github.com/gists/225cbf782c1e4c911dbfb44813adc429
#owner: https://api.github.com/users/jacobbridges

sudo amazon-linux-extras enable emacs

sudo yum update -y

sudo yum install emacs-nox -y

sudo yum install git -y

sudo yum-config-manager --add-repo=https://copr.fedorainfracloud.org/coprs/carlwgeorge/ripgrep/repo/epel-7/carlwgeorge-ripgrep-epel-7.repo

sudo yum install ripgrep

git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf

~/.fzf/install

sudo yum install rust -y

sudo yum install cargo -y

cargo install fd-find

wget https://github.com/tsl0922/ttyd/releases/download/1.7.1/ttyd.x86_64

chmod +x ttyd.x86_64

mkdir -p ~/.local/bin

mv ttyd.x86_64 ~/.local/bin/ttyd

sudo yum install tmux

git clone --depth 1 https://github.com/doomemacs/doomemacs ~/.emacs.d

~/.emacs.d/bin/doom install