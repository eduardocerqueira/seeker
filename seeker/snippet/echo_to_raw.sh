#date: 2022-07-22T17:06:42Z
#url: https://api.github.com/gists/75b250e9d97b6435f5f89f30245ef2a5
#owner: https://api.github.com/users/ichux

# https://docs.docker.com/engine/api/latest/

echo -e "GET /images/json HTTP/1.0\r\n" | nc -U /var/run/docker.sock
echo -e "GET /v1.41/containers/json HTTP/1.0\r\n" | nc -U /var/run/docker.sock

curl -XGET --unix-socket /var/run/docker.sock http://localhost/v1.41/containers/json

curl -s -XPOST --unix-socket /var/run/docker.sock \
	-d '{"Image":"distinctid"}' -H 'Content-Type: application/json' \
	http://localhost/containers/create

# https://man7.org/linux/man-pages/man1/ncat.1.html
ncat -kl -p 2376 -c 'ncat -U /run/user/1000/docker.sock'

# https://kelcecil.com/curl/2015/12/07/query-unix-socket-with-curl.html
