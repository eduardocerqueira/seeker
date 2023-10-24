#date: 2023-10-24T17:08:53Z
#url: https://api.github.com/gists/6e3f824a2c7c28d9af39e5c01a2d6fa3
#owner: https://api.github.com/users/rajathithan

#!/bin/bash
# Installation of Cloud-sql-proxy in DataProc clusters
# Script Re-Modified by - Rajathithan Rajasekar
# Date - October 12, 2023
# projectname:region-name:instance-name
# Reference taken from - https://github.com/GoogleCloudDataproc/initialization-actions/blob/master/cloud-sql-proxy/cloud-sql-proxy.sh 

# Readonly Variables
readonly PROXY_DIR='/var/run/cloud_sql_proxy'
readonly PROXY_BIN='/usr/local/bin/cloud_sql_proxy'
readonly INIT_SCRIPT='/usr/lib/systemd/system/cloud-sql-proxy.service'
readonly PROXY_LOG_DIR='/var/log/cloud-sql-proxy'

 
# Log Function to verify the installation & configuration process
function log() {
  echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] [$(hostname)]: INFO: $*" >&2
}

 
# Proxy-Flags to set the instance name and connection type
# The service account used by the compute engine will have SQL Admin role and SQL admin scopes
# Connection is made via a private ip instead of public
function get_proxy_flags() {
  local proxy_instances_flags=''
  proxy_instances_flags+=" --ip_address_types=PRIVATE"
  proxy_instances_flags+=" -instances=XXPROJECTNAMEXX:XXREGION-NAMEXXX:XXXINSTANCE-NAMEXXX=tcp:3307"
  echo "${proxy_instances_flags}"
}

 
# Installation of the cloud-sql proxy as a service inside the compute engine instance
function install_cloud_sql_proxy() {
  echo 'Installing Cloud SQL Proxy ...' >&2
  # Install proxy.
  wget -nv --timeout=30 --tries=5 --retry-connrefused \
    https://dl.google.com/cloudsql/cloud_sql_proxy.linux.amd64
  mv cloud_sql_proxy.linux.amd64 ${PROXY_BIN}
  chmod +x ${PROXY_BIN}
  mkdir -p ${PROXY_DIR}
  mkdir -p ${PROXY_LOG_DIR}
  local proxy_flags
  proxy_flags="$(get_proxy_flags)"
  # Install proxy as systemd service for reboot tolerance.
  cat <<EOF >${INIT_SCRIPT}
[Unit]
Description=Google Cloud SQL Proxy
After=local-fs.target network-online.target
After=google.service
Before=shutdown.target 

[Service]
Type=simple
ExecStart=/bin/sh -c '${PROXY_BIN} \
  -dir=${PROXY_DIR} \
  ${proxy_flags} >> /var/log/cloud-sql-proxy/cloud-sql-proxy.log 2>&1'

[Install]
WantedBy=multi-user.target

EOF
  chmod a+rw ${INIT_SCRIPT}
  log 'Cloud SQL Proxy installation succeeded'
}

 
# Installation of Additional packages like telnet, mysql-client as required
function install_additional_packages() {
  sed -i 's/http:\/\//https:\/\//g' /etc/apt/sources.list
  log "Installing MySQL CLI & Telnet..."
  if command -v apt >/dev/null; then
    apt update && apt install default-mysql-client -y & apt install telnet -y
  elif command -v yum >/dev/null; then
    yum -y update && yum -y install mysql && yum -y install telnet
  fi
  log "MySQL CLI & Telnet are installed"
}

 
# Start the cloud-sql-proxy service 
function start_cloud_sql_proxy() {
  log 'Starting Cloud SQL proxy ...'
  systemctl enable cloud-sql-proxy
  systemctl start cloud-sql-proxy ||
    err 'Unable to start cloud-sql-proxy service'
  log 'Logs can be found in /var/log/cloud-sql-proxy/cloud-sql-proxy.log'
}


# Installation of dataproc master
function update_master() {
  install_cloud_sql_proxy
  install_additional_packages
  start_cloud_sql_proxy
}

 
# Installation of dataproc worker
function update_worker() {
  install_cloud_sql_proxy
  install_additional_packages
  start_cloud_sql_proxy
}


# Main Function
function main() {
  local role
  role="$(/usr/share/google/get_metadata_value attributes/dataproc-role)"
  if [[ "${role}" == 'Master' ]]; then
    update_master
  else
    update_worker
  fi
  log 'All done'
}

# Call the Main Function
main

 