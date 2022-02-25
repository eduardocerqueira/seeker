#date: 2022-02-25T16:46:00Z
#url: https://api.github.com/gists/1d58f69384f68aaa1300271524cfef7e
#owner: https://api.github.com/users/bschuele

FROM ethereum/client-go:v1.10.1

ARG ACCOUNT_PASSWORD

COPY genesis.json /tmp

RUN geth init /tmp/genesis.json \
    && rm -f ~/.ethereum/geth/nodekey \
    && echo ${ACCOUNT_PASSWORD} > /tmp/password \
    && geth account new --password /tmp/password \
    && rm -f /tmp/password

ENTRYPOINT ["geth"]