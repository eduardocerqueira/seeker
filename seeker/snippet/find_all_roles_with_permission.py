#date: 2023-11-23T16:33:34Z
#url: https://api.github.com/gists/7920acb0e698fb0ccec841f5ec40f67e
#owner: https://api.github.com/users/magnus-longva-bouvet

#!/usr/bin/env python3
import argparse
import itertools
import logging
import re
import sys

from azure.cli.core import get_default_cli


def initialize_logger(log_level=logging.INFO):
    class InfoFilter(logging.Filter):
        def filter(self, rec):
            return rec.levelno in (logging.DEBUG, logging.INFO)

    format_str = '%(asctime)s - %(levelname)s - %(message)s'
    datefmt_str = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(format_str, datefmt_str)
    logger = logging.getLogger("__name__")
    logger.setLevel(log_level)


    h1 = logging.StreamHandler(sys.stdout)
    h1.setFormatter(formatter)
    h1.setLevel(logging.DEBUG)
    h1.addFilter(InfoFilter())

    h2 = logging.StreamHandler()
    h2.setFormatter(formatter)
    h2.setLevel(logging.WARNING)

    logger.addHandler(h1)
    logger.addHandler(h2)
    return logger

def az(args_str):
    args = args_str.split()
    cli = get_default_cli()
    with open("/dev/null", "w") as f:
        try:
            cli.invoke(args, out_file=f)
        except SystemExit as e:
            return ValueError(f"Error executing Azure CLI command: {args_str}. Exit code {e.code}")
        if cli.result.result:
            return cli.result.result
        elif cli.result.error:
            return cli.result.error
        raise Exception("No result or error returned from Azure CLI")


def get_wildcard_permutations_of_permission(permission):
    # we define an empty list to store the wildcard permutations
    wildcard_permutations = []
    # we split the permission on "/"
    permission_parts = permission.split("/")
    # we loop through the permission parts
    for i in range(len(permission_parts)-1):
        # we join the permission parts up to the current index
        wildcard_permutations.append("/".join(permission_parts[:i+1]) + "/*")

    # we return the wildcard permutations
    return wildcard_permutations

def generate_permutations(action):
    # Split the action into segments
    segments = action.split('/')

    # Generate all combinations of segments and '*'
    all_combinations = list(itertools.product(*[(seg, '*') for seg in segments]))

    # Join the segments back with '/'
    permutations = ['/'.join(combo) for combo in all_combinations]

    return permutations


# we define main function
def get_irredundant_matching_permissions(permission):
    matching_permissions = get_wildcard_permutations_of_permission(permission) + [permission] + generate_permutations(
        permission)
    for i in range(len(matching_permissions)):
        matching_permissions[i] = re.sub(r'\*(\/\*)*', '*', matching_permissions[i])

    irredundant_set = [s for s in set(matching_permissions)]

    irredundant_set.sort()
    return irredundant_set


def get_all_roles(logger, subscriptions):
    builtin_roles = az("role definition list")
    custom_roles = []
    for i, sub in enumerate(subscriptions):
        logger.debug(f"Getting custom roles for subscription {sub}, {i}/{len(subscriptions)}")
        custom_roles += az(f"role definition list --subscription {sub}")
    return builtin_roles + custom_roles


def get_matching_roles(logger, roles, irredundant_matching_permissions):
    matching_roles = {}
    for role in roles:
        in_actions = []
        in_not_actions = []
        permissions = role["permissions"]
        for permission in permissions:
            actions = permission["actions"]
            for action in actions:
                if action in irredundant_matching_permissions:
                    in_actions.append(action)
            not_actions = permission["notActions"]
            for not_action in not_actions:
                if not_action in irredundant_matching_permissions:
                    logger.debug(f"notAction {not_action} in {irredundant_matching_permissions}")
                    in_not_actions.append(not_action)
        if in_actions:
            in_actions.sort(key=lambda a: len(a.split("/")))
            longest_match_in_actions = in_actions[-1]
            in_not_actions.sort(key=lambda a: len(a.split("/")))
            longest_match_in_not_actions = in_not_actions[-1] if len(in_not_actions) > 0 else None
            if longest_match_in_not_actions is None or len(longest_match_in_not_actions.split("/")) < len(longest_match_in_actions.split("/")):
                if role["roleName"] not in matching_roles.keys():
                    matching_roles[role["roleName"]] = longest_match_in_actions
            else:
                print(f"notActions included {longest_match_in_not_actions} which is longer than {longest_match_in_actions}")
    return matching_roles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find all role definitions which allow a given resource provider action')
    parser.add_argument('--permission', type=str, required=True, help='The permission to search for in role definitions. Example: Microsoft.Network/virtualNetworks/subnets/join/action')
    parser.add_argument('--log-level', type=str, default="INFO", choices=["DEBUG", "INFO", "WARN", "ERROR", "FATAL"], help="The log level")
    args = parser.parse_args()
    permission_str = args.permission

    logger = initialize_logger(args.log_level)

    irredundant_matching_permissions = get_irredundant_matching_permissions(permission_str)
    subscriptions = az("account list --query [].id --all -o tsv")
    subscriptions.sort()
    roles = get_all_roles(logger, subscriptions)
    matching_roles = get_matching_roles(logger, roles, irredundant_matching_permissions)
    for roleName in matching_roles.keys():
        print(f"{roleName}: {matching_roles[roleName]}")

