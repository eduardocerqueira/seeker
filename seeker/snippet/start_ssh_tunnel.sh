#date: 2023-04-14T17:08:03Z
#url: https://api.github.com/gists/27284f7ec6fcd8151a40a9f146d8385e
#owner: https://api.github.com/users/scriptsandthings

################################################################################
# Creates SSH tunnel between Mac/UNIX-based OS and remote server.
#
# Configure `remote_user` and `remote_host` before using.
#
# Usage:  start_ssh_tunnel.sh
#
################################################################################
# User configurable settings:
remote_user=
remote_host=

# Connection settings:
local_bind_address=127.0.0.1
local_port=3307
destination_host=127.0.0.1
destination_port=3306

# Create SSH tunnel to server
echo "Creating ssh tunnel (press Ctrl + C at any time to disconnect/quit) ..."
ssh -N ${remote_user}@${remote_host} \
  -L ${local_bind_address}:${local_port}:${destination_host}:${destination_port} \
  -o ServerAliveInterval=120
