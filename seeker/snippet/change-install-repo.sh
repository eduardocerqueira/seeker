#date: 2025-10-07T17:03:45Z
#url: https://api.github.com/gists/9fa5e462df692ca77726eef9f0a5a5a1
#owner: https://api.github.com/users/ThinGuy

#!/bin/bash

# Create Directory
mkdir -p ~/www

# Enter directory
cd ~/www

# Create autoinstall.yaml
cat > ~/www/user-data << 'EOF'
#cloud-config
autoinstall:
  apt:
    preserve_sources_list: false
    primary:
      - arches: [amd64]
        uri: http://us.archive.ubuntu.com/ubuntu
    security:
      - arches: [amd64]
        uri: http://us.archive.ubuntu.com/ubuntu        
    fallback: abort
    geoip: true
    sources_list: |
      deb [arch=amd64] $PRIMARY $RELEASE main universe restricted multiverse
      deb [arch=amd64] $PRIMARY $RELEASE-updates main universe restricted multiverse
      deb [arch=amd64] $SECURITY $RELEASE-security main universe restricted multiverse
      deb [arch=amd64] $PRIMARY $RELEASE-backports main universe restricted multiverse
EOF

# Create meta-data file
touch ~/www/meta-data;

# Start webserver to serve file

cd ~/www
python3 -m http.server 3003


# Now you can either provide the URL during install or provide the file.  We will walk you through this.

