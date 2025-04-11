#date: 2025-04-11T16:46:41Z
#url: https://api.github.com/gists/7e842929d864593c08640433a7407885
#owner: https://api.github.com/users/ConnorBaker

#!/usr/bin/env bash

# General script used to set up an Azure HPC instance with Infiniband to use RDMA.
# I do not remember how well this works.

# Set up the UDEV rule for the Mellanox device
# NOTE: Assumes only a single IB interface exists.
echo 'SUBSYSTEM=="net", ACTION=="add", ATTR{dev_id}=="0x0", ATTR{type}=="32", NAME="ib0"' \
    | sudo tee /etc/udev/rules.d/99-rename-ib.rules \
    && sudo udevadm control --reload-rules

# Install the Mellanox OFED drivers

# On names for the mellanox interfaces:
# https://techcommunity.microsoft.com/t5/azure-compute-blog/accelerated-networking-on-hb-hc-hbv2-hbv3-and-ndv2/ba-p/2067965
# For more recent releases of the installer, check:
# https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/
# NOTE: Have not found nfsrdma driver to be available via the DOCA installer
# NOTE: Cannot use --with-nvmf on Ubuntu 24.04 -- install fails.
wget "https://content.mellanox.com/ofed/MLNX_OFED-24.07-0.6.1.0/MLNX_OFED_LINUX-24.07-0.6.1.0-ubuntu24.04-x86_64.tgz" \
    && tar -xvf MLNX_OFED_LINUX-24.07-0.6.1.0-ubuntu24.04-x86_64.tgz \
    && pushd MLNX_OFED_LINUX-24.07-0.6.1.0-ubuntu24.04-x86_64 \
    && sudo ./mlnxofedinstall --with-nfsrdma --force \
    && popd

# Load the drivers and restart Mellanox Software Tools drivers
sudo /etc/init.d/openibd restart \
    && sudo mst restart

# Assign an IP address to the IB interface and bring it up
# NOTE: 10.1.0.4 to match azure-store address on 10.0.2.4
sudo ip addr add 10.1.0.5/16 dev ib0 \
    && sudo ip link set ib0 up

# Install nfs for server
sudo apt install -y nfs-kernel-server

# Drivers we'll need to reload.
# TODO: Remove client/server drivers depending.
cat <<EOF | sudo tee /etc/modules-load.d/rdma.conf
svcrdma
xprtrdma
EOF

# Reload the modules
sudo systemctl restart systemd-modules-load.service

# Format and mount nvme0n1 and nvme1n1 as EXT4
sudo mkfs.ext4 /dev/nvme0n1
sudo mkfs.ext4 /dev/nvme1n1
sudo mkdir -p /drive0 /drive1 /drive0_nfs
sudo mount -t ext4 -o sync,rw /dev/nvme0n1 /drive0
sudo mount -t ext4 -o sync,rw /dev/nvme1n1 /drive1

# Create the NFS export directory
sudo mkdir -p /nix
sudo mount -t tmpfs -o size=300G tmpfs /nix

# Expose /nix via NFS
# https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/configuring_and_using_network_file_services/deploying-an-nfs-server_configuring-and-using-network-file-services#services-required-on-an-nfs-server_deploying-an-nfs-server
# https://infohub.delltechnologies.com/en-us/l/dell-technologies-powerscale-onefs-best-practices-for-davinci-resolve/linux-settings/
echo "/drive0 10.1.0.5(fsid=1,rw,sync,no_wdelay,insecure,no_root_squash,no_subtree_check)" | sudo tee -a /etc/exports

# https://wiki.debian.org/NFSServerSetup
cat <<EOF | sudo tee /etc/default/nfs-kernel-server
RPCNFSDOPTS="-N 2 -N 3 -U --rdma"
RPCMOUNTDOPTS="--manage-gids -N 2 -N 3"
EOF

# https://wiki.debian.org/NFSServerSetup
cat <<EOF | sudo tee /etc/default/nfs-common
NEED_STATD=no
NEED_IDMAPD=yes
NEED_GSSD=no
EOF

# Enable RDMA for NFS
# https://dzone.com/articles/optimizing-infiniband-bandwidth-utilization
# Settings for NFSv4 only:
# https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/7/html/storage_administration_guide/nfs-serverconfig#nfs4-only
cat <<EOF | sudo tee /etc/nfs.conf.d/rdma.conf
[mountd]
threads=64

[nfsd]
threads=64
udp=n
tcp=y
vers3=n
vers4=y
vers4.0=n
vers4.1=n
vers4.2=y
rdma=y
rdma-port=20049
EOF

# Restart the NFS server
sudo systemctl restart nfs-server
sudo exportfs -ra

# On consumers:
# TODO: UDEV rules for consistent naming of the IB interface
sudo ip addr add 10.1.0.5/16 dev ib0

# Bring the interface up
sudo ip link set ib0 up

# Install nfs for client
sudo apt install -y nfs-common

# TODO:
# https://learn.microsoft.com/en-us/azure/azure-netapp-files/performance-linux-concurrency-session-slots#nfsv41
# See "Can increasing session slots increase overall performance?" in the following document:
# https://www.netapp.com/media/10720-tr-4067.pdf
echo "options nfs max_session_slots=180" | sudo tee /etc/modprobe.d/nfsclient.conf

# Mount the NFS share
sudo mkdir -p /nix
# TODO: Optimize parameters
# TODO: nconnect not supported -- is the VAST NFS client required for nconnect option with RDMA?
# TODO: Can't set namelen=1023, not a recognized option?
# Default settings when rsize and wsize are not set (the maximum value, 1MB, is used):
sudo mount -t nfs4 -o rw,sync,noatime,vers=4.2,rsize=1048576,wsize=1048576,namlen=255,acregmin=0,acregmax=0,acdirmin=0,acdirmax=0,hard,noac,proto=rdma,port=20049,nconnect=16,timeo=600,retrans=2,sec=sys,lookupcache=none,local_lock=none 10.1.0.4:/nix /mounted


sudo mount -t nfs4 -o rw,sync,noatime,vers=4.2,rsize=1048576,wsize=1048576,namlen=255,acregmin=0,acregmax=0,acdirmin=0,acdirmax=0,hard,noac,proto=rdma,port=20049,nconnect=16,timeo=600,retrans=2,sec=sys,lookupcache=none,local_lock=none 10.1.0.4:/drive0 /drive0_nfs
