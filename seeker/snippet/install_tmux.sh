#date: 2022-04-22T17:14:00Z
#url: https://api.github.com/gists/7906d4c470a0497b72f9e38756718fc0
#owner: https://api.github.com/users/jcarley

LIBEVENT_VERSION="2.1.12-stable"
TMUX_VERSION="3.2a"

sudo yum install -y gcc kernel-devel make ncurses-devel openssl-devel

curl -LOk https://github.com/libevent/libevent/releases/download/release-${LIBEVENT_VERSION}/libevent-${LIBEVENT_VERSION}.tar.gz
tar -xf libevent-${LIBEVENT_VERSION}.tar.gz
cd libevent-${LIBEVENT_VERSION}
./configure --prefix=/usr/local
make -j4
sudo make -j4 install

cd ../

curl -LOk https://github.com/tmux/tmux/releases/download/${TMUX_VERSION}/tmux-${TMUX_VERSION}.tar.gz
tar -xf tmux-${TMUX_VERSION}.tar.gz
cd tmux-${TMUX_VERSION}
LDFLAGS="-L/usr/local/lib -Wl,-rpath=/usr/local/lib" ./configure --prefix=/usr/local
make -j4
sudo make -j4 install

tmux -V