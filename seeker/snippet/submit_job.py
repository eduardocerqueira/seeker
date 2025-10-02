#date: 2025-10-02T17:07:52Z
#url: https://api.github.com/gists/1649155a776df7da56d41ad0e8ddfabd
#owner: https://api.github.com/users/mstou

# This is adapted for THL2 from https://github.com/epfml/getting-started/blob/main/csub.py
#!/usr/bin/python3

import argparse
from datetime import datetime, timedelta
from pprint import pprint
import re
import subprocess
import tempfile
import yaml
import os

parser = argparse.ArgumentParser(description="Cluster Submit Utility (no-symlinks)")
parser.add_argument(
    "-n", "--name", type=str, required=False,
    help="Job name (has to be unique in the namespace)",
)
parser.add_argument(
    "-c", "--command", type=str, required=False,
    help="Command to run on the instance (default sleep for duration)",
)
parser.add_argument(
    "-t", "--time", type=str, required=False,
    help="The maximum duration allowed for this job (default 7d). "
         "Format like 3d12h30m or 24h or 90m, etc.",
)
parser.add_argument(
    "-g", "--gpus", type=int, default=1, required=False,
    help="The number of GPUs requested (default 1)",
)
parser.add_argument(
    "--cpus", type=int, default=1, required=False,
    help="The number of CPUs requested (default 1)",
)
parser.add_argument(
    "--memory", type=str, default="32G", required=False,
    help="The minimum amount of CPU memory (default 32G). "
         "Must match regex '^([+-]?[0-9.]+)([eEinumkKMGTP]*[-+]?[0-9]*)$'",
)
parser.add_argument(
    "-i", "--image", type=str, required=False,
    default="ic-registry.epfl.ch/mlo/mlo:v1",
    help="The URL of the docker image that will be used for the job",
)
parser.add_argument(
    "-p", "--port", type=int, required=False,
    help="A cluster port for connect to this node (unused by this script)",
)
parser.add_argument(
    "-u", "--user", type=str, default="user.yaml",
    help="Path to a yaml file that defines the user",
)
parser.add_argument(
    "--train", action="store_true",
    help="train job (default is interactive, which has higher priority)",
)
parser.add_argument(
    "-d", "--dry", action="store_true",
    help="Print the generated yaml file instead of submitting it",
)
parser.add_argument(
    "--backofflimit", default=0, type=int,
    help="number of retries before marking a workload as failed (train jobs only; default 0)",
)
parser.add_argument(
    "--node_type", type=str, default="",
    choices=["", "v100", "h100", "h200", "default", "a100-40g"],
    help=("node type to run on (default empty = any). "
          "RCP cluster: use h100 for H100; 'default' for A100-80G interactive; "
          "'a100-40g' for A100-40G; h200 for H200-140GB; v100 for V100-32GB"),
)
parser.add_argument(
    "--host_ipc", action="store_true",
    help="created workload will use the host's ipc namespace",
)
parser.add_argument(
    "--large_shm", action="store_true",
    help="use large shared memory /dev/shm for the job",
)

if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.user):
        print(
            f"User file {args.user} does not exist, use the template in `template/user.yaml` to create your user file."
        )
        exit(1)

    with open(args.user, "r") as file:
        user_cfg = yaml.safe_load(file)

    # the latest version can be found on EPFL wiki (kept from original script)
    runai_cli_version = "2.18.94"
    scratch_name = "thl2-scratch"

    if args.name is None:
        args.name = f"{user_cfg['user']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    if args.time is None:
        args.time = 7 * 24 * 60 * 60
    else:
        pattern = r"((?P<days>\d+)d)?((?P<hours>\d+)h)?((?P<minutes>\d+)m)?((?P<seconds>\d+)s?)?"
        match = re.match(pattern, args.time)
        parts = {k: int(v) for k, v in match.groupdict().items() if v}
        args.time = int(timedelta(**parts).total_seconds())

    if args.command is None:
        args.command = f"sleep {args.time}"

    workload_kind = "TrainingWorkload" if args.train else "InteractiveWorkload"

    working_dir = user_cfg["working_dir"]

    # Build YAML (no symlink envs)
    cfg = f"""
apiVersion: run.ai/v2alpha1
kind: {workload_kind}
metadata:
  annotations:
    runai-cli-version: {runai_cli_version}
  labels:
    PreviousJob: "true"
  name: {args.name}
  namespace: runai-thl2-{user_cfg['user']}
spec:
  name:
    value: {args.name}
  arguments:
    value: "{args.command}"
  environment:
    items:
      HOME:
        value: "/{user_cfg['user']}"
      NB_USER:
        value: {user_cfg['user']}
      NB_UID:
        value: "{user_cfg['uid']}"
      NB_GROUP:
        value: {user_cfg['group']}
      NB_GID:
        value: "{user_cfg['gid']}"
      WORKING_DIR:
        value: "{working_dir}"
      WANDB_API_KEY:
        value: {user_cfg['wandb_api_key']}
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"H "**********"F "**********"_ "**********"T "**********"O "**********"K "**********"E "**********"N "**********": "**********"
        value: "**********"
      EPFML_LDAP:
        value: {user_cfg['user']}
  gpu:
    value: "{args.gpus}"
  cpu:
    value: "{args.cpus}"
  memory:
    value: "{args.memory}"
  image:
    value: {args.image}
  imagePullPolicy:
    value: Always
  pvcs:
    items:
      pvc--0:
        value:
          claimName: {scratch_name}
          existingPvc: true
          path: "/thl2"
          readOnly: false
      pvc--1:
        value:
          claimName: home
          existingPvc: true
          path: "/{user_cfg['user']}"
          readOnly: false
  runAsGid:
    value: {user_cfg['gid']}
  runAsUid:
    value: {user_cfg['uid']}
  runAsUser:
    value: true
  serviceType:
    value: ClusterIP
  username:
    value: {user_cfg['user']}
  allowPrivilegeEscalation:
    value: true
"""

    # Optional additions
    if args.node_type in ["v100", "h100", "h200", "default", "a100-40g"]:
        cfg += f"""
  nodePools:
    value: {args.node_type}
"""
    if args.node_type in ["h100", "default", "a100-40g"] and not args.train:
        cfg += f"""
  preemptible:
    value: true
"""
    if args.host_ipc:
        cfg += f"""
  hostIpc:
    value: true
"""
    if args.train:
        cfg += f"""
  backoffLimit:
    value: {args.backofflimit}
"""
    if args.large_shm:
        cfg += f"""
  largeShm:
    value: true
"""
    
    cfg += f"""
  supplementalGroups:
    value: "{user_cfg['supplemental_groups']}"
"""
    cfg += f"""
  largeShm:
    value: true
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        f.write(cfg)
        f.flush()
        if args.dry:
            print(cfg)
        else:
            result = subprocess.run(
                ["kubectl", "apply", "-f", f.name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if result.returncode != 0:
                print("Error encountered:")
                pprint(result.stderr)
                exit(1)
            else:
                print("Output:")
                print(result.stdout)
                print("If the above says 'created', the job has been submitted.")
                print(
                    f"If the above says 'job unchanged', the job with name {args.name} "
                    f"already exists (and you might need to delete it)."
                )
                print("\nThe following commands may come in handy:")
                print(f"runai exec {args.name} -it zsh  # opens an interactive shell on the pod")
                print(f"runai delete job {args.name}   # kills the job and removes it from the list of jobs")
                print(f"runai describe job {args.name} # shows job status")
                print("runai list jobs                # list all jobs and their status")
                print(f"runai logs {args.name}         # show job logs")
