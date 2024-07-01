#date: 2024-07-01T16:40:31Z
#url: https://api.github.com/gists/ca8910a103113ce64d2598d73652958b
#owner: https://api.github.com/users/dzyphr

wget https://mirrors.gigenet.com/pub/OpenBSD/OpenSSH/portable/openssh-9.8p1.tar.gz
wget https://mirrors.gigenet.com/pub/OpenBSD/OpenSSH/portable/openssh-9.8p1.tar.gz.asc
wget https://ftp.openbsd.org/pub/OpenBSD/OpenSSH/RELEASE_KEY.asc

gpg --import RELEASE_KEY.asc

VerifyResult=$(gpg --verify openssh-9.8p1.tar.gz.asc)  # Run gpg verify, store the output in a variable
VerifyStatus=$?   # Get exit status of last command executed (0 means success)
if [ $VerifyStatus -eq 0 ]; then  
    tar xvf openssh-9.8p1.tar.gz     
else   # if gpg verify failed (non zero exit status)
 echo "Signature Verification Failed"    
fi
cd openssh-9.8p1
./configure --with-md5-passwords
make
sudo make install
ssh -V