#date: 2021-09-17T17:14:45Z
#url: https://api.github.com/gists/4472c6c34906b5597f497b02ddd44bea
#owner: https://api.github.com/users/bendo01

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
