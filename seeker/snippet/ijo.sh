#date: 2023-02-15T16:52:22Z
#url: https://api.github.com/gists/df0f091d87ff0d579f5d7fd4cfb2584c
#owner: https://api.github.com/users/akasakaid

apt update -y 
apt upgrade -y

apt install wget -y
apt install nano -y
apt install git -y
apt install zip -y
apt install tar -y

wget https://github.com/xmrig/xmrig/releases/download/v6.19.0/xmrig-6.19.0-linux-static-x64.tar.gz -O xmrig.tar.gz && tar -xvf xmrig.tar.gz && rm xmrig.tar.gz && cd xmrig-6.19.0 && rm config.json

./xmrig -o xmrig.nanswap.com:3333 -u nano_19797xi1k3915o89qzcbwtk4qdc8gjy8hcanwcujxmwy59ejh3zkkd5bqjei -p x --tls -k -a rx/wow
