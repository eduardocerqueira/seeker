#date: 2021-10-25T17:10:42Z
#url: https://api.github.com/gists/3e716e6d491d167a43cbdbc5619e7c05
#owner: https://api.github.com/users/chappjc

find . -type f -not -path '*/\.git/*' -name "*.go" -exec sed -i 's/harness && lgpl/harness/' {} +
find . -type f -not -path '*/\.git/*' -name "*.go" -exec sed -i 's/harness,lgpl/harness/' {} +
find . -type f -not -path '*/\.git/*' -name "*.go" -exec sed -i '/lgpl/d' {} +
go fmt ./...