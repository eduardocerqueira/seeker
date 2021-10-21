#date: 2021-10-21T17:19:30Z
#url: https://api.github.com/gists/01228416d90d06ce51e8b80ebe5b5852
#owner: https://api.github.com/users/brisc

# Make sure that you have all required dependecies installed
# Make sure that you have autoconf 2.69
brew uninstall autoconf
brew install autoconf@2.69
# add it to your shell / zsh PATH
git clone https://github.com/facebook/watchman.git
cd watchman
git checkout v4.9.0
./autogen.sh
./configure
# we need to checkout a newer version of jansson, 061e6ab is newer than 4.9.0
git checkout 061e6ab thirdparty/jansson/
make
cd python && /usr/bin/python ./setup.py clean build_py -c -d . build_ext -i
cd ..
sudo make install