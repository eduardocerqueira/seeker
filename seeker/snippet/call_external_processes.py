#date: 2023-02-01T16:55:46Z
#url: https://api.github.com/gists/d0cf81cd017f3a5e912350c690df9bb4
#owner: https://api.github.com/users/ichux

import logging
import shlex
import subprocess
from time import sleep

POLL_TIMEOUT = 30


def _run_command(cmd, timeout_secs=300):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    for i in range(timeout_secs):
        sleep(1)
        if proc.poll() is not None:
            (out, _) = proc.communicate()
            return proc, out.decode("utf-8")

    proc.kill()
    return proc, "Timeout of %s secs exceeded." % timeout_secs


def external(data):
    proc, msg = _run_command(shlex.split(data))

    if proc.returncode != 0:
        logging.error(msg)
        return False
    return True


def external_call(data):
    try:
        subprocess.run(
            shlex.split(data), capture_output=True, text=True, check=True, timeout=POLL_TIMEOUT
        )
        return True
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        logging.error(exc)
        return False


if __name__ == '__main__':
    # result = external_call(shlex.join(['bash', '-c', "git clone git@github.com:ichux/modelate.git"]))
    # # result = external_call("bash -c 'git clone git@github.com:ichux/modelate.git'")
    # print(result)

    print(external("git clone git@github.com:ichux/modelate.git"))
