# Code Seeker

# TODO
1. obfuscation sensitive data
2. snippet dentro folder
3. improve report
4. release
5. README (pre-commit run --all)

find code snippets based on [configuration](seeker/seeker.conf)

**life cycle**

find/get -> purge -> push

**report**

* by language
* monitoring specific repo
* by stars
* by period (daily, monthly, yearly)

**code sample**

* by file extension

## Requirements
* GitHub OauthAPI GITHUB_TOKEN
* Python 3

## Setup dev env

```shell
git clone git@github.com:eduardocerqueira/seeker.git
cd seeker
sh ops/scripts/set_dev_env.sh
```

## Build

```shell
sh ops/scripts/egg_build.sh
```

## Install

```shell
# from local build
python3 -m venv venv
source venv/bin/activate
pip install dist/seeker-0.0.1.tar.gz

# from local path with editable
git clone git@github.com:eduardocerqueira/seeker.git
cd seeker
pip install -e .
```

## Run

```shell
export GITHUB_TOKEN=**********
cd seeker
seeker --help
seeker --test
seeker
```

check [report](seeker/report.txt)

## Container

```shell
# build
sh ops/scripts/docker_build.sh
# manually
docker build --build-arg SEEKER_RUN="--test" -t seeker -f Dockerfile . --network host

# run
sh ops/scripts/docker_run.sh
# manually
docker run -e GITHUB_TOKEN=$GITHUB_TOKEN -e GITHUB_USERNAME="eduardocerqueira" -e GITHUB_EMAIL="eduardomcerqueira@gmail.com" -it seeker /bin/bash
```

## Links

* https://gist.github.com/discover
* https://github.com/search?p=2&q=etcd&type=Repositories
* https://docs.github.com/en/rest/reference/repos
* https://pygithub.readthedocs.io/en/latest/
