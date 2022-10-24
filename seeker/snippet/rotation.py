#date: 2022-10-24T17:32:02Z
#url: https://api.github.com/gists/8d029caade45dfb413d335a5df3c892c
#owner: https://api.github.com/users/juftin

#!/usr/bin/env python3

"""
AWS Profile Rotation Script
"""

import configparser
import pathlib
from copy import deepcopy
from typing import Dict, Tuple

import click
from rich import traceback, print

traceback.install(show_locals=True)

aws_dir = pathlib.Path.home().joinpath(".aws").resolve()
config_file = aws_dir.joinpath("config")


def get_parser() -> configparser.ConfigParser:
    """
    Get the parser object
    """
    config_parser = configparser.ConfigParser()
    with config_file.open(mode="r") as config:
        config_parser.read_file(config)
    return config_parser


def get_config_section(
    parser: configparser.ConfigParser, profile: str
) -> Tuple[str, Dict[str, str]]:
    """
    Retrieve a desired section from the config

    Parameters
    ----------
    parser: configparser.ConfigParser
    profile: str

    Returns
    -------
    Tuple[str, Dict[str, str]]
    """
    config_mapping = {
        section: dict(parser.items(section=section)) for section in parser.sections()
    }
    profile_names_cleaned = [
        section_name.replace("profile ", "")
        for section_name in config_mapping.keys()
        if section_name != "profile default"
    ]
    profile_name = f"profile {profile}"
    if profile_name == "profile default":
        raise ValueError(
            "Default Profile Rotation Isn't Supported. "
            f"Available Profiles include {' | '.join(profile_names_cleaned)}"
        )
    if profile_name not in config_mapping:
        raise KeyError(
            f"That profile doesn't exist in the {config_file} file. "
            f"Available Profiles include {' | '.join(profile_names_cleaned)}"
        )
    return profile_name, config_mapping[profile_name]


def set_parser_section(
    parser: configparser.ConfigParser, options: Dict[str, str]
) -> configparser.ConfigParser:
    """
    Set Values on the parser object

    Parameters
    ----------
    parser: configparser.ConfigParser
    options: Dict[str, str]

    Returns
    -------

    """
    new_parser = deepcopy(parser)
    default_profile_name = "profile default"
    if default_profile_name not in new_parser.sections():
        new_parser.add_section(section=default_profile_name)
    for key, value in options.items():
        new_parser.set(section="profile default", option=key, value=value)
    return new_parser


def write_parser_to_file(parser: configparser.ConfigParser) -> pathlib.Path:
    """
    Write the contents of a ConfigParser to the AWS File

    Parameters
    ----------
    parser: configparser.ConfigParser

    Returns
    -------
    pathlib.Path
    """
    with config_file.open(mode="w") as config:
        parser.write(config)


@click.argument("profile")
@click.command("rotate")
def rotate(profile: str) -> None:
    """
    Rotate a listed AWS Profile to your default profile
    """
    parser = get_parser()
    print(f"Rotating to Profile: {profile}")
    _, options = get_config_section(parser=parser, profile=profile)
    parser = set_parser_section(parser=parser, options=options)
    write_parser_to_file(parser=parser)
    print(f"Rotation Complete: {profile}")


if __name__ == "__main__":
    rotate()
