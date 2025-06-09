#date: 2025-06-09T17:12:22Z
#url: https://api.github.com/gists/d80c1946603221666295e7730de78118
#owner: https://api.github.com/users/Yesveer

#!/bin/bash

sudo apt update && sudo apt upgrade -y
sudo apt-get install build-essential fakeroot devscripts equivs mariadb-server python3-mysqldb munge -y
sudo mysql <<EOF
CREATE USER 'slurm'@'localhost' IDENTIFIED BY 'slurm@123';
grant all on slurm_acct_db.* TO 'slurm'@'localhost' identified by 'slurm@123' with grant option;
create database slurm_acct_db;
\q
EOF
echo "Slurm MySQL user and database created successfully."
cd ~/
wget https://download.schedmd.com/slurm/slurm-24.11.5.tar.bz2
tar -xaf slurm-24.11.5.tar.bz2
cd slurm-24.11.5/
sudo mk-build-deps -i debian/control
debuild -b -uc -us
./configure
sudo make install

# Permissions
sudo useradd -r -M -s /bin/false slurm
sudo mkdir /var/spool/slurm
sudo mkdir /var/log/slurm/
sudo touch /var/log/slurm/slurmctld.log
sudo chmod 777 /var/log/slurm/slurmctld.log
sudo chmod 777 /var/spool/slurm
sudo mkdir -p /var/spool/slurm/ctld/statesave
sudo head -c 32 /dev/urandom | base64 | sudo tee /var/spool/slurm/ctld/statesave/jwt_hs256.key > /dev/null
sudo chmod 600 /var/spool/slurm/ctld/statesave/jwt_hs256.key
sudo chown slurm:slurm /var/spool/slurm/ctld/statesave/jwt_hs256.key
sudo useradd --system --no-create-home --shell /usr/sbin/nologin slurmrestd

# Define the target path
CONF_DIR="/usr/local/etc"
CONF_FILE="$CONF_DIR/slurmdbd.conf"

# Create the directory if it doesn't exist
sudo mkdir -p "$CONF_DIR"

# Create and write to the config file
sudo tee "$CONF_FILE" > /dev/null <<EOF
#
# Example slurmdbd.conf file.
#
# See the slurmdbd.conf man page for more information.
#
# Archive info
#ArchiveJobs=yes
#ArchiveDir="/tmp"
#ArchiveSteps=yes
#ArchiveScript=
#JobPurge=12
#StepPurge=1
#
# Authentication info
AuthType=auth/munge
#AuthInfo=/var/run/munge/munge.socket.2
#
# slurmDBD info
DbdAddr=localhost
DbdHost=localhost
#DbdPort=7031
SlurmUser=slurm
#MessageTimeout=300
DebugLevel=verbose
#DefaultQOS=normal,standby
LogFile=/var/log/slurm/slurmdbd.log
PidFile=/var/run/slurmdbd.pid
#PluginDir=/usr/lib/slurm
#PrivateData=accounts,users,usage,jobs
#TrackWCKey=yes
#
# Database info
StorageType=accounting_storage/mysql
StorageHost=localhost
#StoragePort=1234
StoragePass=slurm@123
StorageUser=slurm
StorageLoc=slurm_acct_db
AuthAltTypes=auth/jwt
AuthAltParameters=jwt_key=/var/spool/slurm/ctld/statesave/jwt_hs256.key
EOF

# Set appropriate permissions
sudo chown slurm:slurm "$CONF_FILE"
sudo chmod 600 "$CONF_FILE"
sudo chown slurm:slurm /usr/local/etc/slurmdbd.conf

echo "✅ slurmdbd.conf created at $CONF_FILE"


# Get current hostname
NODE_NAME=$(hostname)

# Define config path
CONFIG_PATH="/usr/local/etc/slurm.conf"

# Create the configuration file with dynamic hostname
cat <<EOF | sudo tee $CONFIG_PATH > /dev/null
ClusterName=single-node-cluster
SlurmctldHost=${NODE_NAME}
MpiDefault=none
ProctrackType=proctrack/linuxproc
ReturnToService=2
SlurmctldPort=6817
SlurmdPort=6818
SlurmdSpoolDir=/var/spool/slurmd
SlurmUser=slurm
StateSaveLocation=/var/spool/slurm
SlurmdPidFile=/var/run/slurmd.pid
SlurmctldPidFile=/var/run/slurmctld.pid
SwitchType=switch/none
TaskPlugin=task/affinity
AccountingStorageType=accounting_storage/slurmdbd
AccountingStorageHost=${NODE_NAME}
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
#DebugLevel=debug
#AuthAltTypes=auth/jwt
#AuthAltParameters=jwt_key=/var/spool/slurm/ctld/statesave/jwt_hs256.key
AuthType=auth/munge
AuthAltTypes=auth/jwt
AuthAltParameters=jwt_key=/var/spool/slurm/ctld/statesave/jwt_hs256.key

# Compute Node
NodeName=${NODE_NAME} CPUs=8 RealMemory=2000 State=UNKNOWN
PartitionName=debug Nodes=ALL Default=YES MaxTime=INFINITE State=UP
EOF

# Set permissions (optional but recommended)
sudo chmod 644 $CONFIG_PATH

echo "✅ slurm.conf created at $CONFIG_PATH with hostname '${NODE_NAME}'"


cd ~/slurm-24.11.5/etc/
sudo cp slurmctld.service.in /etc/systemd/system/slurmctld.service
sudo cp slurmd.service.in /etc/systemd/system/slurmd.service
sudo cp slurmdbd.service.in /etc/systemd/system/slurmdbd.service
sudo cp slurmrestd.service.in /etc/systemd/system/slurmrestd.service
cd ~/
sudo systemctl daemon-reexec
sudo systemctl daemon-reload

sudo systemctl enable slurmctld
sudo systemctl enable slurmd
sudo systemctl enable slurmdbd
sudo systemctl enable slurmrestd

sudo systemctl start slurmctld
SERVICE_FILE="/etc/systemd/system/slurmctld.service"
sudo sed -i 's|@sbindir@/slurmctld|/usr/local/sbin/slurmctld|g' "$SERVICE_FILE"
sudo systemctl start slurmd
SERVICE_FILE="/etc/systemd/system/slurmd.service"
sudo sed -i 's|@sbindir@/slurmd|/usr/local/sbin/slurmd|g' "$SERVICE_FILE"
sudo systemctl start slurmdbd
SERVICE_FILE="/etc/systemd/system/slurmdbd.service"
sudo sed -i 's|@sbindir@/slurmdbd|/usr/local/sbin/slurmdbd|g' "$SERVICE_FILE"
sudo systemctl start slurmrestd
SERVICE_FILE="/etc/systemd/system/slurmrestd.service"
sudo sed -i 's|@sbindir@/slurmrestd|/usr/local/sbin/slurmrestd|g' "$SERVICE_FILE"
sudo sed -i 's|@SLURMRESTD_PORT@|6820|g'  "$SERVICE_FILE"
if ! grep -q "^User=slurm" "$SERVICE_FILE"; then
  sudo sed -i '/^\[Service\]/a User=slurmrestd' "$SERVICE_FILE"
  echo "✅ Added 'User=slurm' to $SERVICE_FILE"
else
  echo "ℹ️ User=slurm already present in $SERVICE_FILE"
fi

if ! grep -q "^Group=slurm" "$SERVICE_FILE"; then
  sudo sed -i '/^\[Service\]/a Group=slurmrestd' "$SERVICE_FILE"
  echo "✅ Added 'Group=slurmrestd' to $SERVICE_FILE"
else
  echo "ℹ️ Group=slurm already present in $SERVICE_FILE"
fi

sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl restart slurmctld slurmctld slurmd slurmdbd slurmrestd

echo "✅ Slurm services started successfully."
echo "Ready to use Slurm on this single-node setup."
echo " Ready to go make crazy things with Slurm! 🚀"
echo "Happy Hacking"