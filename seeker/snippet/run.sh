#date: 2021-09-24T17:13:26Z
#url: https://api.github.com/gists/190ba0079b2ce2e2cf385e5030019721
#owner: https://api.github.com/users/micahyoung

docker build --tag rsync -f <(echo "FROM alpine\nRUN apk update && apk add rsync") /var/empty

rsync -vv -e 'docker run -v foo:/foo -i' -a rsync:/foo