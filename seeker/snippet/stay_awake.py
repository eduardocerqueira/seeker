#date: 2021-11-12T17:17:28Z
#url: https://api.github.com/gists/900548d9826b50994ebd42aaa1e6d16f
#owner: https://api.github.com/users/hirak99

"""Stays awake as long as this script is active.

Requires -
sudo apt install xprintidle xdotool

Based on -
https://askubuntu.com/questions/524384/how-can-i-keep-the-computer-awake-depending-on-activity

"""

import logging
import subprocess
import time

 # Number of seconds to start preventing blank screen / suspend.
_IDLE_SECS = 120


def main():
  while True:
      curr_idle = int(subprocess.check_output(['xprintidle']).decode('utf-8'))
      logging.info('Idle for %d ms', curr_idle)
      if int(curr_idle) > _IDLE_SECS * 1000:
          subprocess.call(['xdotool', 'key', 'Control_L'])
          logging.info('Sent activity')
      time.sleep(10)

if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  main()
