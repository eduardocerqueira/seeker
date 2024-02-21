#date: 2024-02-21T17:07:09Z
#url: https://api.github.com/gists/727d5fa393259deb050a7feaccf2b090
#owner: https://api.github.com/users/m0zgen

echo "Please enter your golang path (ex: $HOME/golang) :"
read gopath

echo "Please enter your github username (ex: vsouza) :"
read user


mkdir $gopath
mkdir -p $gopath/src/github.com/$user

export GOPATH=$gopath
export GOROOT=/usr/local/opt/go/libexec
export PATH=$PATH:$GOPATH/bin
export PATH=$PATH:$GOROOT/bin

/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew update
brew install go
brew install git

go get golang.org/x/tools/cmd/godoc
go get golang.org/x/tools/cmd/vet
