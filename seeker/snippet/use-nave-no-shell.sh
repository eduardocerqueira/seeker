#date: 2024-05-16T16:51:34Z
#url: https://api.github.com/gists/2de7300048d74c367aadb587f8e81fee
#owner: https://api.github.com/users/janarthanancs

# use nave, but without a subshell
# note that we're doing "usemain" instead of "use"

mkdir ~/.nave
cd ~/.nave
wget http://github.com/isaacs/nave/raw/master/nave.sh
sudo ln -s $PWD/nave.sh /usr/local/bin/nave

# take ownership
# please don't do this if you don't know what it does!
sudo mkdir -p /usr/local/{share/man,bin,lib/node,include/node}
sudo chown -R $USER /usr/local/{share/man,bin,lib/node,include/node}

# install the latest stable nodejs in the "main" root.
nave usemain stable

curl https://www.npmjs.org/install.sh | sh
