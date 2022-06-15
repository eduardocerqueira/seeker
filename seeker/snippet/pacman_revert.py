#date: 2022-06-15T17:03:58Z
#url: https://api.github.com/gists/021ad961214b0ab38e707e4bacc146f3
#owner: https://api.github.com/users/TheIdealis

#!/usr/bin/python

from subprocess import call
import re
from os.path import exists
from os import environ

home = environ['HOME']
reg_time = re.compile(r'\[(\d{4}.+\d{4})\]')

dates = []

with open('/var/log/pacman.log', 'r')  as f:
  for line in f.readlines():
    timestemp = reg_time.findall(line)
    if timestemp:
      date = timestemp[0].split('T')[0]
      if dates:
        if dates[-1] != date:
          dates.append(date)
      else:
        dates.append(date)
      # Don't use time yet
      # time = timestemp[0].split('T')[1].split('+')[0]

# Print last 10 dates
dates = dates[::-1]
print('Please choose the update you want to revert to:')
for i, date in enumerate(dates[:10]):
  print(i, ' - ', date)

print('')
choice = input("Number: ")

date = dates[int(choice)]

call(f"grep -a {date} /var/log/pacman.log | grep upgraded > ~/.local/tmp/upgraded_pkg.txt", shell=True)
call("awk '{print $4}' ~/.local/tmp/upgraded_pkg.txt  > ~/.local/tmp/pkg_base.txt", shell=True)
call("awk '{print $5}' ~/.local/tmp/upgraded_pkg.txt | sed 's/(/-/g' > ~/.local/tmp/pkg_version.txt", shell=True)
call("paste ~/.local/tmp/pkg_base.txt ~/.local/tmp/pkg_version.txt > ~/.local/tmp/upgraded_pkg.txt.txt", shell=True)
call('tr -d "[:blank:]" < ~/.local/tmp/upgraded_pkg.txt.txt > ~/.local/tmp/pkg.txt', shell=True)

filenames = []
with open(home + '/.local/tmp/pkg.txt', 'r') as f:
  for filename in f.readlines():
    filename = filename.strip()
    if exists("/var/cache/pacman/pkg/"+filename+"-x86_64.pkg.tar.zst"):
      filename = "/var/cache/pacman/pkg/"+filename+"-x86_64.pkg.tar.zst"
    else:
      filename = "/var/cache/pacman/pkg/"+filename+"-any.pkg.tar.zst"
    filenames.append(filename)

call("sudo pacman -U " + ' '.join(filenames), shell=True)
# for i in $(cat ~/pkg.txt);do sudo pacman --noconfirm -U "$i"-x86_64.pkg.tar.zst; done
