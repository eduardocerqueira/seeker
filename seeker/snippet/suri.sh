#date: 2021-11-29T17:15:23Z
#url: https://api.github.com/gists/45394169317d57db352672ec7d541538
#owner: https://api.github.com/users/Taraj

sudo apt-get update
sudo apt-get install libyaml-dev libjansson-dev libcap-dev libpcap.dev rustc cargo -y
cd Downloads/
tar xvzf suricata-6.0.4.tar.gz suricata-6.0.4
cd suricata-6.0.4
./configure
sudo make
sudo make install-full