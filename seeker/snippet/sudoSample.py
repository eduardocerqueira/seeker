#date: 2022-04-28T17:11:49Z
#url: https://api.github.com/gists/846af8756b4ff02452a11bcb8bfdeffa
#owner: https://api.github.com/users/serguei9090

import subprocess

# run command using Popen and provide password using communicate.
# password requires byte format.
# sudo -S brings password request to PIPE
proc = subprocess.Popen(['sudo', '-S', 'nano', 'test.txt'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate(input=b'password\n')

# Print output and errors
print(proc)
