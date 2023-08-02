#date: 2023-08-02T16:49:02Z
#url: https://api.github.com/gists/99218291dd64ff835872b0d971acac32
#owner: https://api.github.com/users/VietThan

# inspiration from https://github.com/themotte/rDrama/commit/9da3451ed97ec62cf4dc403e2863ae7d2fdfba98
# import pprint
import logging
import subprocess

from collections.abc import Callable

LOGGER = logging.getLogger(__name__)


def _execute(command: list[str], check = True, on_stdout_line: Callable[[str]] = None, on_stderr_line: Callable[[str]] = None) -> subprocess.CompletedProcess:
    """Execute command in a `subprocess`

    Parameters
    ----------
    command : list[str]
        example: ["echo", "hello"]
    check : bool, optional
        check if return code of subprocess is not 0 and raise `CalledProcessError`, by default True
    on_stdout_line : Callable[[str]], optional
        a callable to process each line of `stdout`, by default None
    on_stderr_line : Callable[[str]], optional
         a callable to process on each line of `stderr`, by default None

    Returns
    -------
    subprocess.CompletedProcess

    Raises
    ------
    subprocess.CalledProcessError
        if returncode from subprocess is not 0 instead of returning `subprocess.CompletedProcess`. 
        To suppress, set `check` to False.
    """
    LOGGER.debug(f'_execute command: {command}')

    with subprocess.Popen(
        command,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as proc:
        stdout = None
        if proc.stdout:
            stdout = ''
            for line in proc.stdout:
                if on_stdout_line:
                    on_stdout_line(line)
                stdout += line

        stderr = None
        if proc.stderr:
            stderr = ''
            for line in proc.stderr:
                if on_stderr_line:
                    on_stderr_line(line)
                stderr += line

        proc.wait()
        if check and proc.returncode != 0:
            LOGGER.debug(f'STDOUT:\n{stdout}')
            LOGGER.debug(f'STDERR:\n{stderr}')

            raise subprocess.CalledProcessError(
                proc.returncode, command, stdout or None, stderr or None
            )
        else:
            return subprocess.CompletedProcess(
                command, proc.returncode, stdout or None, stderr or None
            )


def _start_all(will_build: bool = True) -> subprocess.CompletedProcess:
    LOGGER.info('Starting containers . . .')
    LOGGER.info(
        '  This can take a while.'
    )
    command = [
        'docker',
        'compose',
        '-f',
        'docker-compose.yaml',
        'up',
        '-d',
    ]
    if will_build:
        LOGGER.info(
            '  Building is invoked.'
        )
        command.append('--build')

    result = _execute(command)

    # alright this seems sketchy, bear with me

    # previous versions of this code used the '--wait' command-line flag
    # the problem with --wait is that it waits for the container to be healthy and working
    # "but wait, isn't that what we want?"
    # ah, but see, if the container will *never* be healthy and working - say, if there's a flaw causing it to fail on startup - it just waits forever
    # so that's not actually useful

    # previous versions of this code also had a check to see if the containers started up properly
    # but this is surprisingly annoying to do if we don't know the containers' names
    # docker-compose *can* do it, but you either have to use very new features that aren't supported on Ubuntu 22.04, or you have to go through a bunch of parsing pain
    # and it kind of doesn't seem necessary

    # see, docker-compose in this form *will* wait until it's *attempted* to start each container.
    # so at this point in execution, either the containers are running, or they're crashed
    # if they're running, hey, problem solved, we're good
    # if they're crashed, y'know what, problem still solved! because our next command will fail

    # maybe there's still a race condition? I dunno! Keep an eye on this.
    # If there is a race condition then you're stuck doing something gnarly with `docker-compose ps`. Good luck!

    LOGGER.info('  Containers started!')

    return result


def _stop_all() -> subprocess.CompletedProcess:
    # use "stop" instead of "down" to avoid killing all stored data
    command = ['docker', 'compose', 'stop']
    LOGGER.info('Stopping all containers . . .')
    result = _execute(command)
    return result