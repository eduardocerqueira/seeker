#date: 2024-12-24T17:04:58Z
#url: https://api.github.com/gists/72c3b839c262325d7caf01427b4623a7
#owner: https://api.github.com/users/idylicaro

wget https://dl.google.com/go/go1.23.0.linux-amd64.tar.gz
sudo tar -xvf go1.23.0.linux-amd64.tar.gz
sudo mv go /usr/local
echo "export GOROOT=/usr/local/go" >> ~/.zshrc
echo "export GOPATH=\$HOME/go" >> ~/.zshrc
echo "export PATH=\$GOPATH/bin:\$GOROOT/bin:\$PATH" >> ~/.zshrc
source ~/.zshrc