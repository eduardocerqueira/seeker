#date: 2022-06-03T16:51:57Z
#url: https://api.github.com/gists/8ed1e0569a22a167320bd0ff0b6781cf
#owner: https://api.github.com/users/Speculative

git clone --bare git@github.com:Speculative/dotfiles.git $HOME/.dotfiles
git --git-dir="$HOME/.dotfiles/" --work-tree="$HOME" config --local status.showUntrackedFiles no
echo "alias dotfiles='/usr/bin/git --git-dir=$HOME/.dotfiles/ --work-tree=$HOME'" >> $HOME/.bashrc