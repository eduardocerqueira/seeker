#date: 2022-04-06T17:06:38Z
#url: https://api.github.com/gists/b68b0d7ecf8ec35a4d1ee7a9ea8552b8
#owner: https://api.github.com/users/gwisp2

FROM hertzg/rtl_433:latest

RUN apk add --no-cache eudev 
COPY --chmod=755 entrypoint.sh /opt/

ENTRYPOINT '/opt/entrypoint.sh'
