# seeker

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

## Build

```shell
sh ops/scripts/egg_build.sh

# or manually
python3 -m pip install --upgrade build
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
export TOKEN=**********
cd seeker
python main.py
```

check [report](seeker/report.txt)

## Container

```shell
# build
sh ops/scripts/docker_build.sh

# run
docker run -e TOKEN=$TOKEN -e GITHUB_USERNAME="eduardomcerqueira" -e GITHUB_EMAIL="eduardomcerqueira@gmail.com" -it seeker /bin/bash
```

## Links

* https://gist.github.com/discover
* https://github.com/search?p=2&q=etcd&type=Repositories
* https://docs.github.com/en/rest/reference/repos
* https://pygithub.readthedocs.io/en/latest/