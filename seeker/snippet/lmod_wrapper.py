#date: 2022-04-14T17:06:07Z
#url: https://api.github.com/gists/85842b675777d7e3ba3337e7a67156d2
#owner: https://api.github.com/users/parsa

import os
import subprocess


def lmod(command: str, *args: str, verbose: bool = False, dry_run: bool = False):
    """
    Basic wrapper for lmod commands.
    Based on https://github.com/TACC/Lmod/blob/master/init/env_modules_python.py.in

    Parameters
    ----------
    command : str
        The command to run.
    args : list
        The args to pass to the lmod command.
    verbose : bool
        Whether to print the lmod output before executing them. Default is False.
    dry_run : bool
        Whether to print the lmod command instead of executing it. Default is False.

    Examples
    --------
    >>> lmod("list")
    >>> lmod("load", "gcc/11.2.0")
    >>> lmod("load", "gcc cmake")
    >>> lmod("load", "gcc cmake ninja", verbose=True)
    """
    proc = subprocess.run(
        [os.environ["LMOD_CMD"], "python", command] + list(args),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print(proc.stderr)

    if verbose:
        print(proc.stdout)

    proc.check_returncode()

    if not dry_run:
        exec(proc.stdout)


lmod("load", "gcc", verbose=True)
