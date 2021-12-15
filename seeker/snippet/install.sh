#date: 2021-12-15T16:59:50Z
#url: https://api.github.com/gists/3a5df055ebc6cff9d382523e93e0cbf4
#owner: https://api.github.com/users/sistlm

# Modified from the gist @https://gist.github.com/odiumediae/3b22d09b62e9acb7788baf6fdbb77cf8

sudo apt-get remove -y --purge vim vim-runtime vim-gnome vim-tiny vim-gui-common
 
sudo apt-get install -y liblua5.1-dev luajit libluajit-5.1 python-dev ruby-dev libperl-dev libncurses5-dev libatk1.0-dev libx11-dev libxpm-dev libxt-dev

#Optional: so vim can be uninstalled again via `dpkg -r vim`
sudo apt-get install -y checkinstall

sudo rm -rf /usr/local/share/vim /usr/bin/vim

cd ~
git clone https://github.com/vim/vim
cd vim
git pull && git fetch

#In case Vim was already installed
cd src
make distclean
cd ..

 ./configure \
 --enable-multibyte \
 --enable-perlinterp=dynamic \
 --enable-rubyinterp=dynamic \
 --with-ruby-command=/usr/bin/ruby \
 --enable-pythoninterp=dynamic \
 --with-python-config-dir=/usr/lib/python2.7/config-arm-linux-gnueabihf \
 --enable-python3interp \
 --with-python3-config-dir=/usr/lib/python3.5/config-3.5m-arm-linux-gnueabihf \
 --enable-luainterp \
 --with-luajit \
 --enable-cscope \
 --enable-gui=auto \
 --with-features=huge \
 --with-x \
 --enable-fontset \
 --enable-largefile \
 --disable-netbeans \
 --with-compiledby="ngs" \
 --enable-fail-if-missing

make && sudo make install