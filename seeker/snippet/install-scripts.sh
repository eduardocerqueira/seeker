#date: 2021-09-21T16:52:09Z
#url: https://api.github.com/gists/e8d0699189b87ef3cad22909b98439be
#owner: https://api.github.com/users/Voker2311

#!/bin/bash

apt install python3-pip
apt install python
apt install gobuster
wget https://github.com/ffuf/ffuf/releases/download/v1.3.1/ffuf_1.3.1_linux_amd64.tar.gz -O /usr/local/bin/ffuf
go install github.com/hakluke/hakrawler
go get github.com/hakluke/hakrevdns
pip3 install dnsgen
go get -u github.com/tomnomnom/httprobe
GO111MODULE=on go get -v github.com/projectdiscovery/httpx/cmd/httpx
git clone https://github.com/blechschmidt/massdns.git
cd massdns
make
cp massdns /usr/local/bin/
cd ..
GO111MODULE=on go get -v github.com/projectdiscovery/shuffledns/cmd/shuffledns
git clone https://github.com/codingo/DNSCewl.git
cd DNSCewl
make
cp DNScewl /usr/local/bin/
cd ..
GO111MODULE=on go get -u -v github.com/lc/gau
go get github.com/tomnomnom/waybackurls
pip3 install uro
go get -u github.com/tomnomnom/anew
wget https://github.com/Findomain/Findomain/releases/download/5.0.0/findomain-linux -O findomain
mv findomain /usr/local/bin/
GO111MODULE=on go get -v github.com/projectdiscovery/subfinder/v2/cmd/subfinder
GO111MODULE=on go get -u -v github.com/lc/subjs
go get -u github.com/tomnomnom/assetfinder
apt install -y libpcap-dev
GO111MODULE=on go get -v github.com/projectdiscovery/naabu/v2/cmd/naabu
git clone https://github.com/ozguralp/gmapsapiscanner.git
GO111MODULE=on go get -v github.com/projectdiscovery/nuclei/v2/cmd/nuclei
cp /root/go/bin/* /usr/local/bin/