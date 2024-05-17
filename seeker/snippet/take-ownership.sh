#date: 2024-05-16T16:51:34Z
#url: https://api.github.com/gists/2de7300048d74c367aadb587f8e81fee
#owner: https://api.github.com/users/janarthanancs

# take ownership of the folders that npm/node use
# please don't do this if you don't know what it does!
sudo mkdir -p /usr/local/{share/man,bin,lib/node,include/node}
sudo chown -R $USER /usr/local/{share/man,bin,lib/node,include/node}

# now just a pretty vanilla node install
# let it use the default paths, but don't use sudo, since there's no need
mkdir node-install
curl http://nodejs.org/dist/node-v0.4.3.tar.gz | tar -xzf - -C node-install
cd node-install/*
./configure
make install

# now the npm easy-install
curl https://www.npmjs.org/install.sh | sh