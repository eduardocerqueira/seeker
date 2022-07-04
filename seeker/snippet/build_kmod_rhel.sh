#date: 2022-07-04T17:04:42Z
#url: https://api.github.com/gists/b4cf2dc96b022841f9677ecd344f3264
#owner: https://api.github.com/users/sshimko

#!/bin/bash
# Builds a module that is present in the kernel sources but not built and shipped by Red Hat.
# For me, this was a 5Gbps USB NIC needing the aqc111 module (not atlantic) on RHEL 8.
#
# It grabs the kernel src RPM based on the latest you have installed
# This might be a lot easier in DKMS.
MODULE="aqc111"
MODULE_KPATH="drivers/net/usb"
MODULE_KCONF="CONFIG_USB_NET_AQC111"

set -x
set -e

# /bin/dnf update kernel kernel-core kernel-headers kernel-devel kernel-modules
# Get the version info for the latest installed
k=$(/bin/rpm -q kernel|rpmdev-sort|tail -n 1)
v=$(/bin/rpm -q --queryformat '%{VERSION}' ${k})
r=$(/bin/rpm -q --queryformat '%{RELEASE}' ${k})
d=$(/bin/rpm --eval '%{dist}')
u=$(/bin/uname -m)

# snag the relevant src RPM and "install" it
/bin/dnf download --source kernel-${v}-${r}
/bin/rpm -ivh kernel-${v}-${r}.src.rpm

# unpack and run the build prep step to apply patches etc
cd $(/bin/rpm --eval '%{_topdir}')
/bin/rpmbuild -bp kernel.spec

# now we can build our module
# sed nastiness b/c the subdir is named using a var/macro defined in the spec, e.g. RELEASE will be *el8_6 but this needs to be *el8 (dist)
# referring to the spec turns out to be similarly ugly and less intuitive to read (nested calls to rpm --eval)
cd ../BUILD/kernel-${v}-${r}/linux-${v}-$(echo ${r} | sed -e "s/\(.*${d}\).*/\1.${u}/")/drivers/net/usb
/bin/make -C /usr/src/kernels/${v}-${r}.${u} SUBDIRS=$(/bin/pwd) M=$(/bin/pwd) CONFIG_RETPOLINE=n ${MODULE_KCONF}=m ${MODULE}.ko

# install our module
/bin/sudo /bin/cp ${MODULE}.ko /lib/modules/${v}-${r}.${u}/kernel/${MODULE_KPATH}
/bin/sudo /sbin/depmod -a ${v}-${r}.${u}