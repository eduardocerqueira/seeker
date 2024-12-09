#date: 2024-12-09T17:11:27Z
#url: https://api.github.com/gists/7eddd504cc3d444f6a01c32d49ac2890
#owner: https://api.github.com/users/Scofield626

# Typically, perf can be installed with
# sudo apt install linux-tools-common linux-tools-generic linux-tools-`uname -r`
# in some cases, when linux-tools-'uname -r' cannot be found for a given kernel, try build from src as below.

# install dependencies
sudo apt-get install -y build-essential git flex bison
sudo apt install -y libzstd1 libdwarf-dev libdw-dev binutils-dev libcap-dev libelf-dev libnuma-dev python3 python3-dev python-setuptools libssl-dev libunwind-dev libdwarf-dev zlib1g-dev liblzma-dev libaio-dev libtraceevent-dev debuginfod libpfm4-dev libslang2-dev systemtap-sdt-dev libperl-dev binutils-dev libbabeltrace-dev libiberty-dev libzstd-dev

git clone --depth 1 https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git
cd linux/tools/perf
make
cp perf /usr/bin