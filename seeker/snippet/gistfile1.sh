#date: 2021-12-15T17:08:48Z
#url: https://api.github.com/gists/e9e4065b96d4a35399a164d7d4504c36
#owner: https://api.github.com/users/sistlm

# This is how you add a Jenkins slave

# On master:

  sudo -u jenkins -H ssh-keygen

# On slave

  adduser --system --group --home=/var/lib/jenkins-slave --no-create-home --disabled-password --quiet --shell /bin/bash jenkins-slave
  install -d -o jenkins-slave -g jenkins-slave /var/lib/jenkins-slave
  install -d -m 700 -o jenkins-slave -g jenkins-slave /var/lib/jenkins-slave/.ssh
  vi /var/lib/jenkins-slave/.ssh/authorized_keys
    copy jenkinsmasterhost:/var/lib/jenkins/ssh/id_rsa.pub

# On master

  sudo -u jenkins -H ssh jenkins-slave@muskatas
    # remember the host key

# Now you can add an SSH slave through the Jenkins web UI, it'll download all the Java .jars automatically
# (you may need to sudo apt-get install openjdk or something on the slave machine)

# To allow the slave to get access to git repos on mygitserver, run this on the slave machine:

  sudo -u jenkins-slave -H ssh-keygen

# On mygitserver

  adduser --system --group --home=/var/lib/jenkins-slave --no-create-home --disabled-password --quiet --shell /usr/bin/git-shell jenkins-slave
  install -d -o jenkins-slave -g jenkins-slave /var/lib/jenkins-slave
  install -d -m 700 -o jenkins-slave -g jenkins-slave /var/lib/jenkins-slave/.ssh
  vi /var/lib/jenkins-slave/.ssh/authorized_keys
    copy /var/lib/jenkins-slave/ssh/id_rsa.pub from the slave machine

# On slave

  sudo -u jenkins-slave -H ssh mygitserver
    # remember the host key
  sudo -u jenkins-slave -H git config --global user.name 'Jenkins'
  sudo -u jenkins-slave -H git config --global user.email 'root@example.com'
