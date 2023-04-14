#date: 2023-04-14T17:08:03Z
#url: https://api.github.com/gists/27284f7ec6fcd8151a40a9f146d8385e
#owner: https://api.github.com/users/scriptsandthings

################################################################################
# Creates SSH tunnel between Mac/UNIX-based OS in background, and runs Stata.
#
# Configure `remote_user`, `remote_host`, `odbc_lib_path` and `stata_path` 
# before using.
#
# Usage:  stata_odbc_mac.sh
#
# Note: SSH tunnel will be terminated on exiting Stata.
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

# Configure LD_LIBRARY_PATH for Apple Silicon or Intel-based Macs:
  # Uncomment for Intel-based Macs:
  odbc_lib_path=/usr/local/Cellar/:/usr/local/Cellar/mariadb-connector-odbc/

  # Uncomment for Apple Silicon Macs
  # odbc_lib_path=/opt/homebrew/lib/:/opt/homebrew/mariadb-connector-odbc/

# Stata Path
stata_path=/Applications/Stata/StataIC.app/Contents/MacOS/StataIC
# stata_path=/Applications/Stata/StataSE.app/Contents/MacOS/StataSE
# stata_path=/Applications/Stata/StataMP.app/Contents/MacOS/StataMP

# Create SSH tunnel to server, configure library path and launch Stata
ssh -fN -L ${local_bind_address}:${local_port}:${destination_host}:${destination_port} ${remote_user}@${remote_host}
export LD_LIBRARY_PATH=${odbc_lib_path} &&
  "${stata_path}" && \
  pkill -f "ssh -fN -L ${local_bind_address}:${local_port}:${destination_host}:${destination_port} ${remote_user}@${remote_host}"
