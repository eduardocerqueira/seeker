#date: 2022-06-17T17:09:05Z
#url: https://api.github.com/gists/af9399475861bc48cd2164ccaba8b4ce
#owner: https://api.github.com/users/katrbhach

import json

with open("/vault/secrets/config.json") as fi:
    repl_config = json.load(fi)


def does_env_exist(environment):
    return environment in repl_config


def get_api_interface(environment):
    return repl_config[environment]["api"]


def get_replicator_connect_url(environment):
    return repl_config[environment]["replicator_connect_url"]


def get_kafka_details(environment):
    return repl_config[environment]["kafka"]
