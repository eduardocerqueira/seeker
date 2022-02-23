#date: 2022-02-23T16:55:53Z
#url: https://api.github.com/gists/ba0b4061b42da5fed2b0fa26b7e1b0df
#owner: https://api.github.com/users/fjmatos

cd /tmp/
wget https://kasm-static-content.s3.amazonaws.com/kasm_release_1.10.0.238225.tar.gz 
tar -xf kasm_release_1.10.0.238225.tar.gz

# Editar  requirements para incluir  ubuntu 21.10
sudo /tmp/kasm/install.sh

