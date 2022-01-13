#date: 2022-01-13T17:06:34Z
#url: https://api.github.com/gists/b8bd71a01a024e589c5c08f166657867
#owner: https://api.github.com/users/simleo

#!/usr/bin/env python

import re
import subprocess as sp

import docker


PATTERN = re.compile("^GPU\s+(\d+):.+UUID:\s+([A-Za-z0-9-]+)")


def map_gpu_ids():
    rval = {}
    out = sp.check_output("nvidia-smi -L", shell=True, universal_newlines=True)
    for line in out.splitlines():
        m = PATTERN.match(line)
        if m:
            idx, id_ = m.groups()
            rval[idx] = id_
    return rval


def map_container_gpus(client):
    rval = {}
    for c in client.containers.list():
        v = "all"
        for kv in c.attrs["Config"]["Env"]:
            try:
                k, v = kv.split("=", 1)
            except ValueError:
                continue
            if k == "NVIDIA_VISIBLE_DEVICES":
                break
        gpus = v.split(",")
        if gpus == ["all"]:
            continue
        for g in gpus:
            rval.setdefault(g, []).append(c)
    return rval



def main():
    gpu_ids = map_gpu_ids()
    container_gpus = map_container_gpus(docker.from_env())
    for idx, id_ in gpu_ids.items():
        print(f"GPU {idx}:")
        try:
            containers = container_gpus[id_]
        except KeyError:
            continue
        for c in containers:
            print(f"  [{c.short_id}] {c.name}")


if __name__ == "__main__":
    main()
