#date: 2022-08-01T17:16:20Z
#url: https://api.github.com/gists/eb105e2b59d2778d90bc58366821928b
#owner: https://api.github.com/users/sohag1426

# Uncomment the next line if you have installed wkhtmltopdf
# sudo apt remove wkhtmltopdf
cd ~
# Select an appropriate link for your system (32 or 64 bit) from the page https://wkhtmltopdf.org/downloads.html and past to the next line
wget https://github.com/wkhtmltopdf/wkhtmltopdf/releases/download/0.12.4/wkhtmltox-0.12.4_linux-generic-amd64.tar.xz
tar xvf wkhtmltox*.tar.xz
sudo mv wkhtmltox/bin/wkhtmlto* /usr/bin
sudo apt-get install -y openssl build-essential libssl-dev libxrender-dev git-core libx11-dev libxext-dev libfontconfig1-dev libfreetype6-dev fontconfig