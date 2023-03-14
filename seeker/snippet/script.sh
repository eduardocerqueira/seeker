#date: 2023-03-14T16:52:56Z
#url: https://api.github.com/gists/919729c3c89bc39dbff648f9f5dae200
#owner: https://api.github.com/users/suhailroushan13

sudo apt update && sudo apt upgrade -y
sudo apt install -y git
sudo apt install net-tools -y
sudo apt-get -y install nginx -y
sudo apt install snapd -y
sudo snap install core 
sudo apt-get install python3-certbot-nginx -y
sudo snap install --classic certbot
sudo ln -s /snap/bin/certbot /usr/bin/
sudo apt install python3-pip -y
sudo apt-get autoremove -y
sudo apt full-upgrade -y
sudo apt install gcc -y
git config --global user.name "suhailroushan13"
git config --global user.email suhailroushan13@gmail.com
curl -fsSL https://deb.nodesource.com/setup_current.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo npm install -g n 
sudo npm install -g npm@9.1.3 
sudo apt install unzip -y
sudo npm i -g pm2 -y
curl -fsSL https://pkg.jenkins.io/debian-stable/jenkins.io.key | sudo tee \
    /usr/share/keyrings/jenkins-keyring.asc > /dev/null
echo deb [signed-by=/usr/share/keyrings/jenkins-keyring.asc] \
    https://pkg.jenkins.io/debian-stable binary/ | sudo tee \
    /etc/apt/sources.list.d/jenkins.list > /dev/null
sudo apt-get install fontconfig openjdk-11-jre -y
sudo apt-get update -y
sudo apt-get install jenkins -y
 yes "" | sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install build-essential
sudo apt-get install manpages-dev
sudo apt install g++
