#date: 2025-02-26T17:11:29Z
#url: https://api.github.com/gists/df77406a00d60705fa5c727b82f36b3e
#owner: https://api.github.com/users/jeyaramashok

sudo rm -rf \
/usr/local/{lib/node{,/.npm,_modules},bin,share/man}/{npm*,node*,man1/node*}
sudo rm -rf \
  /usr/local/bin/npm \
  /usr/local/share/man/man1/node* \
  /usr/local/lib/dtrace/node.d \
  ~/.npm \
  ~/.node-gyp
sudo rm -rf /opt/local/bin/node /opt/local/include/node /opt/local/lib/node_modules
sudo rm -rf /usr/local/bin/npm /usr/local/share/man/man1/node.1 /usr/local/lib/dtrace/node.d