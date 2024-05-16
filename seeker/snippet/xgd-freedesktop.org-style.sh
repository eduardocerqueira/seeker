#date: 2024-05-16T16:51:34Z
#url: https://api.github.com/gists/2de7300048d74c367aadb587f8e81fee
#owner: https://api.github.com/users/janarthanancs

echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
echo 'export npm_config_userconfig=$HOME/.config/npmrc' >> ~/.bashrc
. ~/.bashrc
mkdir ~/.local
mkdir ~/node-latest-install
cd ~/node-latest-install
curl http://nodejs.org/dist/node-latest.tar.gz | tar xz --strip-components=1
./configure --prefix=~/.local
make install
curl https://www.npmjs.org/install.sh | sh
