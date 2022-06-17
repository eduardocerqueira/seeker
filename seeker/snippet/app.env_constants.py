#date: 2022-06-17T17:02:31Z
#url: https://api.github.com/gists/471bc58e08abcddfb2d6229a40b5aa9a
#owner: https://api.github.com/users/katrbhach

from os import environ


class EnvironmentConstants(object):

    env_name: str = environ.get("ENV_NAME")
    url: str = environ.get("URL")
    username: str = environ.get("USERNAME")
    password: str = environ.get("PASSWORD")
    acl_api_url: str = environ.get("ACL_API_URL")
    audit_kafka_bootstrap_servers: str = environ.get("AUDIT_KAFKA_BOOTSTRAP_SERVERS")
    audit_kafka_username: str = environ.get("AUDIT_KAFKA_USERNAME")
    audit_kafka_password: str = environ.get("AUDIT_KAFKA_PASSWORD")
    audit_schema_reg_url: str = environ.get("AUDIT_SCHEMA_REGISTRY_URL")
    audit_schema_reg_username: str = environ.get("AUDIT_SCHEMA_REGISTRY_USERNAME")
    audit_schema_reg_password: str = environ.get("AUDIT_SCHEMA_REGISTRY_PASSWORD")
