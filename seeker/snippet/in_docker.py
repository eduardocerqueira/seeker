#date: 2025-07-29T17:01:21Z
#url: https://api.github.com/gists/81e51013f20798de3028c0e6a077a63b
#owner: https://api.github.com/users/mslanker

'''
Copyright (C) 2018 by Daniel Foose

Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE
'''

import re
import docker
import socket


def in_docker():
    """
    Check if running in Docker
    :return: Whether or not this is running in Docker container
    """
    try:
        with open('/proc/1/cgroup', 'rt') as cgroup_file:
            return 'docker' in cgroup_file.read()
    except FileNotFoundError as e:
        return False


def translate_path(path):
    """
    :param path: A string representing a path within the container
    :return: A string representing a path on the host (or the original path if the path is not in a bound volume)
    """
    binds = get_binds()
    if path in binds.keys():
        return binds[path]
    exps = ['(%s)/(.*)' % key for key in binds.keys()]
    print('path: %s' % path)
    for exp in exps:
        result = re.search(exp, path)
        if result:
            print('%s/%s' % (binds[result.group(1)], result.group(2)))
            return '%s/%s' % (binds[result.group(1)], result.group(2))
    raise ValueError('Path %s not present in a bind mount. Volume mount will fail when running this in Docker.' % path)


def get_current_container():
    """
    Will raise ValueError if there is no container with the same hostname as the environment this is running in
    Which indicates that this is not a docker container, or that /var/run/docker.sock is not bind mounted to
    /var/run/docker.sock on the host (i.e. this is a container which is also a docker host).
    :return: A dictionary containing information about the container this is running in obtained using docker api
    """
    hostname = socket.gethostname()
    client = docker.from_env()
    for container in client.containers.list():
        if container.attrs['Config']['Hostname'] == hostname:
            return container
    raise ValueError('Not running in Docker container')


def get_binds():
    """
    :return: A dictionary with paths in the container as keys and paths on the host as values
    """
    container = get_current_container()
    return {bind.split(':')[1]: bind.split(':')[0] for bind in container.attrs['HostConfig']['Binds']}