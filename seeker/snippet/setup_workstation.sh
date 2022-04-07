#date: 2022-04-07T17:09:43Z
#url: https://api.github.com/gists/f8c9424d1295139d7a0bc3a8a0b8f9d4
#owner: https://api.github.com/users/kadencartwright

git clone --bare https://github.com/kadencartwright/dotfiles.git $HOME/.cfg
function config {
   /usr/bin/git --git-dir=$HOME/.cfg/ --work-tree=$HOME $@
}
mkdir -p .config-backup
config checkout
if [ $? = 0 ]; then
  echo "Checked out config.";
  else
    echo "Backing up pre-existing dot files.";
    config checkout 2>&1 | egrep "\s+\." | awk {'print $1'} | xargs -I{} mv {} .config-backup/{}
fi;
config checkout
config config status.showUntrackedFiles no