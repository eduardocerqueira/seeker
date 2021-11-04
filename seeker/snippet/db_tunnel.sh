#date: 2021-11-04T16:56:59Z
#url: https://api.github.com/gists/cf6c99085146b31a58c5a1b3e7a66524
#owner: https://api.github.com/users/kmuenkel

ssh -f -N -J {username}@{jump.domain} {username}@{target.domain} -L {local-port:=3306}:localhost:3306
#DB_HOST=host.docker.internal