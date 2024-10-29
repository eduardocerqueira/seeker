#date: 2024-10-29T16:56:59Z
#url: https://api.github.com/gists/f417732927a0fbd6f3ff79b181146aeb
#owner: https://api.github.com/users/jnhmcknight

"""
Include this file in your project, and then import it instead of the real `boto3`,
wherever you need to create a `boto3.client` or `boto3.resource`

i.e.:

import wrapped_boto3

s3 = wrapped_boto3.client('s3')

"""

import os
import boto3 as real_boto3
from flask import current_app


# Add any boto3 keyword argument names here, with their app/env var name.
APP_OVERRIDES = {
    'endpoint_url': 'AWS_ENDPOINT_URL',
}


def add_app_aws_config(kwargs):
    for k,v in APP_OVERRIDES.items():
        try:
            config_value = current_app.config[v] or os.environ.get(v)
            if not kwargs.get(k) and config_value:
                kwargs.update({
                    k: config_value,
                })
        except (IndexError, AttributeError):
            pass

    return kwargs


def client(*args, **kwargs):
    kwargs = add_app_aws_config(kwargs)
    return real_boto3.client(*args, **kwargs)


def resource(*args, **kwargs):
    kwargs = add_app_aws_config(kwargs)
    return real_boto3.resource(*args, **kwargs)