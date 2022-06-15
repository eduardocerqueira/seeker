#date: 2022-06-15T17:06:46Z
#url: https://api.github.com/gists/89da1e241cf351e40210deeb585725a4
#owner: https://api.github.com/users/Elimuhub-coder

#!/bin/sh

# Install Golang 1.4 on Amazon Linux

echo "install correct language pack"
cat <<EOF > /etc/default/locale
LANG=en_US.UTF-8
LANGUAGE=en_US
LC_CTYPE=en_US.UTF-8
LC_ALL=en_US.UTF-8
EOF
. /etc/default/locale

echo "install Mercurial"
yum install mercurial -y

echo "download Go and install it, as well as create GOPATH directory"
cd ~

wget https://storage.googleapis.com/golang/go1.4.2.linux-amd64.tar.gz
tar -xf go1.4.2.linux-amd64.tar.gz && rm go1.4.2.linux-amd64.tar.gz
mv go /usr/local && mkdir -p /usr/local/gopath

echo "set enviornment variables required for Go"
export GOROOT=/usr/local/go
export GOPATH=/usr/local/gopath
cat <<EOF >> /etc/profile.d/dev-env.sh
export GOROOT=/usr/local/go
export GOPATH=/usr/local/gopath
export "PATH=/usr/local/gopath/bin:/usr/local/go/bin:/opt/rubies/ruby-2.1.4/bin:/sbin:/bin:/usr/sbin:/usr/bin:/opt/aws/bin:/opt:$PATH"
export GORACE=log_path=/usr/local/gopath/racereport
export w=/usr/local/gopath/src/github.com
EOF
. /etc/profile.d/dev-env.sh

# install Go tools
echo "installing go tool ... golint"
go get github.com/golang/lint/golint
echo "installing go tool ... errcheck"
go get github.com/kisielk/errcheck
echo "installing go tool ... benchcmp"
go get golang.org/x/tools/cmd/benchcmp
echo "installing go tool ... impl"
go get github.com/josharian/impl
echo "installing go tool ... goimports"
go get golang.org/x/tools/cmd/goimports
echo "installing go tool ... goreturns"
go get sourcegraph.com/sqs/goreturns
echo "installing go tool ... godef"
go get code.google.com/p/rog-go/exp/cmd/godef
echo "installing go tool ... gocode"
go get github.com/nsf/gocode
echo "installing go tool ... pq"
go get github.com/lib/pq
echo "installing go tool ... gorename"
go get golang.org/x/tools/cmd/gorename
echo "installing go tool ... godepgraph"
go get github.com/kisielk/godepgraph

echo "install done."
