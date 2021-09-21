#date: 2021-09-21T17:14:20Z
#url: https://api.github.com/gists/63b930f9c360018cec3bfae7d72c61b5
#owner: https://api.github.com/users/AbbasHMB

After login as root...

# Create a new user
adduser demo

# Should only belong to the [username] group
groups demo 

# Add demo to the sudo group
usermod -a -G sudo demo

# Copy your ssh-key to the ssh keys of this new user so we can start logging in as them
ssh-copy-id -i ~/.ssh/id_rsa.pub demo@192.1.1.1

# Install Git so we can install our server
apt-get install git

# Download the script
git clone git://github.com/Xeoncross/lowendscript.git /root/git/

# Setup locals
dpkg-reconfigure locales

# Run it
cd /root/git
./setup-debian.sh dotdeb
./setup-debian.sh system
./setup-debian.sh dropbear 22
./setup-debian.sh iptables 22
./setup-debian.sh nginx
./setup-debian.sh php
./setup-debian.sh mysql

# Now create a site
./setup-debian.sh site example.com

# You can push a folder from your computer now 
sudo rsync -av --stats --progress --rsh='ssh -p22' ./ demo@192.1.1.1:~/foldername
cp -R foldername /var/www/example.com/public
rm foldername

# Or use git to checkout something
cd /var/www/example.com/public
git clone [URL] ./


