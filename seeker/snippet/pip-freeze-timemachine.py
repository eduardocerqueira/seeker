#date: 2024-10-29T16:51:30Z
#url: https://api.github.com/gists/738113004ef1f4155a012d49feb31f93
#owner: https://api.github.com/users/u8sand

#!/usr/bin/env python
'''
This script installs packages from requirements file(s) in a virtual environment
 as they would have been installed at a certian date, then assembles the versions
 for the requirements.txt file.

Usage:
  pip-freeze-timemachine -d 2022-01-01 -r requirements.txt -o requirements.versioned.txt

requirements:
  sh
  click
  pypi-timemachine
'''

import sh
import re
import sys
import time
import click
import socket
import shutil
import pathlib
import tempfile
import contextlib
from datetime import datetime

def find_free_port():
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.bind(('localhost', 0))
  port = sock.getsockname()[1]
  sock.close()
  return port

def wait_for_port(port: int, host: str = 'localhost', poll_interval = 0.1, timeout: float = 5.0):
  start_time = time.perf_counter()
  while True:
    try:
      with socket.create_connection((host, port), timeout=timeout):
        break
    except ConnectionRefusedError as ex:
      time.sleep(poll_interval)
      if time.perf_counter() - start_time >= timeout:
        raise TimeoutError(f"Waiting for port {host}:{port}") from ex

def decode_datetime(value):
  try:
    return datetime.strptime(value, r'%Y-%m-%d')
  except ValueError:
    return datetime.strptime(value, r'%Y-%m-%dT%H:%M:%S')

def encode_datetime(dt):
  return datetime.strftime(dt, r'%Y-%m-%dT%H:%M:%S')

def validate_iso(ctx, param, value):
  try:
    return decode_datetime(value)
  except ValueError:
    raise click.BadParameter('Expected YYYY-MM-DD[THH:MM:SS]')

@click.command()
@click.option('-d', '--cutoff-date', type=click.UNPROCESSED, callback=validate_iso, default=encode_datetime(datetime.now()))
@click.option('-r', '--requirement', type=click.File('r'), multiple=True, default=[sys.stdin])
@click.option('-o', '--output', type=click.File('w'), default=sys.stdout)
def main(cutoff_date, requirement, output):
  free_port = find_free_port()
  sh.Command(sys.executable)('--version', _out=sys.stderr, _err=sys.stderr)
  time_machine_proc = sh.Command(sys.executable)('-m', 'pypi_timemachine', encode_datetime(cutoff_date), f"--port={free_port}", _bg=True, _bg_exc=False)
  try:
    wait_for_port(free_port)
    tmpd = tempfile.mkdtemp()
    try:
      requirements_input = '\n'.join([reqs.read() for reqs in requirement])
      pathlib.Path(f"{tmpd}/requirements.txt").write_text(requirements_input)
      sh.Command(sys.executable)('-m', 'venv', f"{tmpd}/venv", _out=sys.stderr, _err=sys.stderr)
      sh.Command(f"{tmpd}/venv/bin/python")('-m', 'pip', 'install', '-r', f"{tmpd}/requirements.txt", _out=sys.stderr, _err=sys.stderr)
      freeze_output = sh.Command(f"{tmpd}/venv/bin/python")('-m', 'pip', 'freeze', _err=sys.stderr)
      requirement_packages = {
        pkgmatch.group(1).lower(): pkgmatch
        for pkgspec in requirements_input.splitlines()
        for pkgmatch in (re.match(r'^(\w+)[^@]*?(@.+)?$', pkgspec),)
        if pkgmatch
      }
      output.writelines([
        f"{requirement_pkgmatch.group(0)}\n" if requirement_postfix else f"{pkgspec}\n"
        for pkgspec in freeze_output.splitlines()
        for pkgmatch in (re.match(r'^(\w+)', pkgspec),)
        if pkgmatch
        for requirement_pkgmatch in (requirement_packages.get(pkgmatch.group(1).lower()),)
        if requirement_pkgmatch
        for requirement_postfix in (requirement_pkgmatch.group(2),)
      ])
    finally:
      shutil.rmtree(tmpd)
  finally:
    with contextlib.suppress(sh.SignalException_SIGTERM):
      time_machine_proc.terminate()
      time_machine_proc.wait()

if __name__ == '__main__':
  main()