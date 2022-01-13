#date: 2022-01-13T17:13:44Z
#url: https://api.github.com/gists/ce5a1c2e8a21f821d4d19afd10d04195
#owner: https://api.github.com/users/prashant-shahi

# uname -a
# Linux ip-172-31-1-128 5.4.0-1038-aws #40-Ubuntu SMP Fri Feb 5 23:53:34 UTC 2021 aarch64 aarch64 aarch64 GNU/Linux

git clone --depth=1 --branch=docker_server_from_ci_builds https://github.com/filimonov/ClickHouse.git 

cd ClickHouse/docker/server/
docker build . --network host --build-arg single_binary_location="https://builds.clickhouse.tech/master/aarch64/clickhouse"
docker image ls
docker tag 98f169cda25a altinity/clickhouse-server:21.4.1.6307-testing-arm
docker login
docker push altinity/clickhouse-server:21.4.1.6307-testing-arm